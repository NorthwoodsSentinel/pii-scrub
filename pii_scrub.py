#!/usr/bin/env python3
"""
pii-scrub — Detect, pseudonymize, and restore PII in text.

Python port of jcfischer/pii-pseudonymizer (TypeScript/Bun).
Adds transcript-aware chunking for meeting recordings and call transcripts.

Detection layers:
  - Pattern-based: email, phone, IP, URL, domain (regex)
  - NER-based: person names, organizations, locations (spaCy)

Pseudonymization:
  - Consistent fake replacements via Faker (same input = same output)
  - Session files for round-trip restore (pseudonymize → process → restore)

Usage:
    pii_scrub.py detect   [--file FILE] [--no-ner] [--transcript]
    pii_scrub.py pseudo   [--file FILE] [--save-session FILE] [--load-session FILE] [-v]
    pii_scrub.py pseudo   --transcript [--file FILE] [--save-session FILE] [-v]
    pii_scrub.py restore  [--file FILE] --load-session FILE

    cat transcript.txt | python3 pii_scrub.py pseudo --transcript -v
    echo "John met Sarah at Acme Corp" | python3 pii_scrub.py pseudo -v

Requires:
    pip install spacy faker
    python3 -m spacy download en_core_web_sm
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy imports — spaCy and faker are heavy, only load when needed
# ---------------------------------------------------------------------------

_nlp = None
_faker = None


def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("ERROR: spaCy model not found. Run: python3 -m spacy download en_core_web_sm",
                  file=sys.stderr)
            sys.exit(1)
    return _nlp


def get_faker():
    global _faker
    if _faker is None:
        from faker import Faker
        _faker = Faker()
    return _faker


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ENTITY_TYPES = {"PERSON", "ORG", "LOCATION", "EMAIL", "PHONE", "IP", "URL", "DOMAIN"}

# spaCy label → our type
SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
}


@dataclass
class Entity:
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0
    method: str = "pattern"


@dataclass
class Mapping:
    original: str
    pseudonym: str
    type: str


@dataclass
class Session:
    session_id: str
    mappings: dict = field(default_factory=dict)       # original → Mapping
    reverse: dict = field(default_factory=dict)         # pseudonym → original

    def add(self, original: str, pseudonym: str, entity_type: str):
        if original in self.mappings:
            return
        self.mappings[original] = Mapping(original, pseudonym, entity_type)
        self.reverse[pseudonym] = original

    def get_pseudonym(self, original: str) -> Optional[str]:
        m = self.mappings.get(original)
        return m.pseudonym if m else None

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "mappings": [
                {"original": m.original, "pseudonym": m.pseudonym, "type": m.type}
                for m in self.mappings.values()
            ],
        }

    @classmethod
    def from_dict(cls, data):
        s = cls(session_id=data["session_id"])
        for m in data.get("mappings", []):
            s.add(m["original"], m["pseudonym"], m["type"])
        return s

    def save(self, path: str):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Session":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

PATTERNS = [
    ("EMAIL",  re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')),
    ("URL",    re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')),
    ("IP",     re.compile(r'\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b')),
    ("PHONE",  re.compile(r'(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{2,4}[-.\s]?\d{2,4}(?:[-.\s]?\d{2,4})?')),
    ("DOMAIN", re.compile(r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b')),
]

MIN_PHONE_DIGITS = 10


def _is_valid_ip(text: str) -> bool:
    parts = text.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def _is_valid_phone(text: str) -> bool:
    digits = re.sub(r'\D', '', text)
    return len(digits) >= MIN_PHONE_DIGITS


def detect_patterns(text: str) -> list[Entity]:
    entities = []
    covered = []

    for etype, regex in PATTERNS:
        for m in regex.finditer(text):
            start, end = m.start(), m.end()
            matched = m.group(0)

            if any(start < ce and end > cs for cs, ce in covered):
                continue

            if etype == "IP" and not _is_valid_ip(matched):
                continue
            if etype == "PHONE" and not _is_valid_phone(matched):
                continue
            if etype == "DOMAIN":
                if start > 0 and text[start - 1] == "@":
                    continue
                if start >= 3 and text[start - 3:start] == "://":
                    continue

            entities.append(Entity(matched, etype, start, end, 1.0, "pattern"))
            covered.append((start, end))

    return entities


# ---------------------------------------------------------------------------
# NER detection via spaCy
# ---------------------------------------------------------------------------

# Timestamp pattern that spaCy misidentifies as entities
_TIMESTAMP_RE = re.compile(r'^\d{2}:\d{2}:\d{2}$')

# Minimum length for PERSON/LOCATION entities (filters single-char noise)
_NER_MIN_LENGTH = 3

# Override this set to add domain-specific terms that should never be pseudonymized.
# Example: NER_SKIP_TERMS.update({"Kubernetes", "Terraform", "RBAC"})
NER_SKIP_TERMS: set[str] = set()


def _should_skip_ner_entity(text: str, etype: str) -> bool:
    stripped = text.strip()
    if _TIMESTAMP_RE.match(stripped):
        return True
    if stripped in NER_SKIP_TERMS:
        return True
    if re.match(r'^[\d:\s\n]+$', stripped):
        return True
    if len(stripped) < _NER_MIN_LENGTH and etype in ("PERSON", "LOCATION"):
        return True
    if '\n' in stripped:
        return True
    if ',' in stripped:
        return True
    return False


def detect_ner(text: str) -> list[Entity]:
    nlp = get_nlp()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        our_type = SPACY_LABEL_MAP.get(ent.label_)
        if our_type is None:
            continue
        if _should_skip_ner_entity(ent.text, our_type):
            continue
        entities.append(Entity(
            text=ent.text,
            type=our_type,
            start=ent.start_char,
            end=ent.end_char,
            confidence=0.95,
            method="ner",
        ))
    return entities


# ---------------------------------------------------------------------------
# Transcript chunker
# ---------------------------------------------------------------------------

# Matches lines like "00:01:32 John Moser" or "00:01:32 Speaker 3"
_TRANSCRIPT_TS_RE = re.compile(r'^(\d{2}:\d{2}:\d{2})\s+(.+)$')


def chunk_transcript(text: str, max_lines: int = 15) -> list[str]:
    """Split transcript into speaker-turn blocks for NER processing.

    Splits on timestamp lines (HH:MM:SS Speaker). Falls back to fixed-size
    line blocks if no timestamps are found.
    """
    lines = text.split('\n')
    chunks = []
    current = []

    for line in lines:
        if _TRANSCRIPT_TS_RE.match(line):
            if current:
                chunks.append('\n'.join(current))
                current = []
        current.append(line)

    if current:
        chunks.append('\n'.join(current))

    # Fallback: chunk by line count if no timestamps found
    if len(chunks) <= 1 and len(lines) > max_lines * 2:
        chunks = []
        for i in range(0, len(lines), max_lines):
            chunk = '\n'.join(lines[i:i + max_lines])
            if chunk.strip():
                chunks.append(chunk)

    return chunks if chunks else [text]


# ---------------------------------------------------------------------------
# Unified detection
# ---------------------------------------------------------------------------

def detect_all(text: str, use_ner: bool = True, transcript_mode: bool = False) -> list[Entity]:
    """Run all detectors. In transcript mode, chunk first then merge."""

    if transcript_mode and use_ner:
        chunks = chunk_transcript(text)
        all_entities = []
        offset = 0

        for chunk in chunks:
            idx = text.find(chunk, offset)
            if idx == -1:
                idx = offset
            chunk_offset = idx

            for e in detect_patterns(chunk):
                e.start += chunk_offset
                e.end += chunk_offset
                all_entities.append(e)

            for e in detect_ner(chunk):
                e.start += chunk_offset
                e.end += chunk_offset
                all_entities.append(e)

            offset = chunk_offset + len(chunk)

        return _deduplicate(all_entities)

    entities = detect_patterns(text)
    if use_ner:
        entities.extend(detect_ner(text))
    return _deduplicate(entities)


def _deduplicate(entities: list[Entity]) -> list[Entity]:
    """Remove duplicate/overlapping entities. Prefer longer span, then higher confidence."""
    entities.sort(key=lambda e: (e.start, -(e.end - e.start), -e.confidence))
    result = []
    last_end = -1
    for e in entities:
        if e.start >= last_end:
            result.append(e)
            last_end = e.end
        elif e.end - e.start > (result[-1].end - result[-1].start) if result else 0:
            result[-1] = e
            last_end = e.end
    return result


# ---------------------------------------------------------------------------
# Pseudonym generator (seeded Faker for consistency)
# ---------------------------------------------------------------------------

def _seed_hash(entity_type: str, original: str) -> int:
    h = hashlib.sha256(f"{entity_type}:{original}".encode()).hexdigest()
    return int(h[:8], 16)


def generate_pseudonym(entity_type: str, original: str) -> str:
    fake = get_faker()
    fake.seed_instance(_seed_hash(entity_type, original))

    generators = {
        "PERSON": fake.name,
        "ORG": fake.company,
        "LOCATION": fake.city,
        "EMAIL": fake.email,
        "PHONE": fake.phone_number,
        "IP": fake.ipv4,
        "URL": fake.url,
        "DOMAIN": fake.domain_name,
    }
    gen = generators.get(entity_type)
    return gen() if gen else f"[REDACTED_{entity_type}]"


# ---------------------------------------------------------------------------
# Pseudonymize
# ---------------------------------------------------------------------------

def pseudonymize(text: str, session: Session, use_ner: bool = True,
                 transcript_mode: bool = False) -> tuple[str, list[Entity]]:
    entities = detect_all(text, use_ner=use_ner, transcript_mode=transcript_mode)

    if not entities:
        return text, []

    sorted_ents = sorted(entities, key=lambda e: e.start, reverse=True)
    result = text

    for e in sorted_ents:
        pseudonym = session.get_pseudonym(e.text)
        if not pseudonym:
            pseudonym = generate_pseudonym(e.type, e.text)
            session.add(e.text, pseudonym, e.type)
        result = result[:e.start] + pseudonym + result[e.end:]

    return result, entities


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------

def restore(text: str, session: Session) -> tuple[str, int]:
    count = 0
    pseudonyms = sorted(session.reverse.keys(), key=len, reverse=True)
    for pseudo in pseudonyms:
        original = session.reverse[pseudo]
        while pseudo in text:
            text = text.replace(pseudo, original, 1)
            count += 1
    return text, count


# ---------------------------------------------------------------------------
# Session ID generator
# ---------------------------------------------------------------------------

def make_session_id() -> str:
    ts = hex(int(time.time()))[2:]
    rand = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    return f"pii_{ts}_{rand}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def read_input(args) -> str:
    if hasattr(args, 'file') and args.file:
        return Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif not sys.stdin.isatty():
        return sys.stdin.read()
    else:
        print("ERROR: No input. Provide --file or pipe stdin.", file=sys.stderr)
        sys.exit(1)


def cmd_detect(args):
    text = read_input(args)
    entities = detect_all(text, use_ner=not args.no_ner, transcript_mode=args.transcript)
    output = {
        "entities": [
            {"text": e.text, "type": e.type, "start": e.start, "end": e.end,
             "confidence": e.confidence, "method": e.method}
            for e in entities
        ],
        "count": len(entities),
    }
    print(json.dumps(output, indent=2))


def cmd_pseudo(args):
    text = read_input(args)

    if args.load_session:
        session = Session.load(args.load_session)
    else:
        session = Session(session_id=make_session_id())

    result, entities = pseudonymize(
        text, session, use_ner=not args.no_ner, transcript_mode=args.transcript
    )

    if args.save_session:
        session.save(args.save_session)

    if args.verbose:
        output = {
            "text": result,
            "sessionId": session.session_id,
            "replacementCount": len(entities),
            "entities": [{"type": e.type, "original": e.text} for e in entities],
            "translationTable": [
                {"original": m.original, "pseudonym": m.pseudonym, "type": m.type}
                for m in session.mappings.values()
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(result)


def cmd_restore(args):
    text = read_input(args)
    if not args.load_session:
        print("ERROR: --load-session required for restore.", file=sys.stderr)
        sys.exit(1)
    session = Session.load(args.load_session)
    result, count = restore(text, session)
    print(result)
    print(f"Restored {count} replacements.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="pii-scrub -- Detect, pseudonymize, and restore PII in text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # detect
    p_detect = sub.add_parser("detect", help="Detect PII entities in text")
    p_detect.add_argument("--file", "-f", help="Input file (default: stdin)")
    p_detect.add_argument("--no-ner", action="store_true", help="Disable NER, pattern-only")
    p_detect.add_argument("--transcript", "-t", action="store_true", help="Transcript chunking mode")

    # pseudo
    p_pseudo = sub.add_parser("pseudo", help="Pseudonymize PII in text")
    p_pseudo.add_argument("--file", "-f", help="Input file (default: stdin)")
    p_pseudo.add_argument("--save-session", "-s", help="Save session mapping to file")
    p_pseudo.add_argument("--load-session", "-l", help="Load existing session mapping")
    p_pseudo.add_argument("--no-ner", action="store_true", help="Disable NER, pattern-only")
    p_pseudo.add_argument("--transcript", "-t", action="store_true", help="Transcript chunking mode")
    p_pseudo.add_argument("--verbose", "-v", action="store_true", help="Output JSON with translation table")

    # restore
    p_restore = sub.add_parser("restore", help="Restore pseudonyms to originals")
    p_restore.add_argument("--file", "-f", help="Input file (default: stdin)")
    p_restore.add_argument("--load-session", "-l", required=True, help="Session mapping file")

    args = parser.parse_args()

    if args.command == "detect":
        cmd_detect(args)
    elif args.command == "pseudo":
        cmd_pseudo(args)
    elif args.command == "restore":
        cmd_restore(args)


if __name__ == "__main__":
    main()
