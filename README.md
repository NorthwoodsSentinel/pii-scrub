# pii-scrub

Detect, pseudonymize, and restore personally identifiable information (PII) in text. Built for processing meeting transcripts, call recordings, and documents before sending to AI services.

Python port of [jcfischer/pii-pseudonymizer](https://github.com/jcfischer/pii-pseudonymizer) by Jens-Christian Fischer (TypeScript/Bun). This version adds transcript-aware chunking for speaker-turn formats and runs on standard Python with spaCy.

## What it does

Replaces real PII with consistent fake data powered by Faker. The same input always produces the same pseudonym, so "John Smith" becomes "Ashley Jackson" everywhere in the document. Save the session file and you can restore the originals later.

**Detection layers:**
- **Pattern-based** — email, phone, IPv4, URL, domain (regex)
- **NER-based** — person names, organizations, locations (spaCy `en_core_web_sm`)

**Transcript mode** splits timestamped transcripts (`HH:MM:SS Speaker Name`) into speaker-turn chunks before running NER. This solves a known limitation where NER models return zero entities on long-form transcript input.

## Install

```bash
pip install spacy faker
python3 -m spacy download en_core_web_sm
```

No other dependencies. Python 3.10+.

## Usage

### Pseudonymize text

```bash
echo "Robert met James at Acme Corp in Madison." | python3 pii_scrub.py pseudo -v
```

### Pseudonymize a transcript

```bash
python3 pii_scrub.py pseudo --transcript --file meeting.txt --save-session session.json -v
```

### Restore originals

```bash
python3 pii_scrub.py restore --file pseudonymized.txt --load-session session.json
```

### Detect only (no replacement)

```bash
python3 pii_scrub.py detect --file document.txt
python3 pii_scrub.py detect --transcript --file transcript.txt
```

### Pattern-only mode (no NER)

```bash
echo "email me at john@example.com" | python3 pii_scrub.py pseudo --no-ner -v
```

## Filtering false positives

NER models sometimes tag technical terms, acronyms, or product names as entities. Use the `NER_SKIP_TERMS` set to suppress these:

```python
from pii_scrub import NER_SKIP_TERMS, pseudonymize, Session, make_session_id

NER_SKIP_TERMS.update({
    "Kubernetes", "Terraform", "RBAC", "Prisma Access",
    "AnyConnect", "Zscaler", "GlobalProtect",
})

session = Session(session_id=make_session_id())
result, entities = pseudonymize(text, session, transcript_mode=True)
```

## Session files

Session files are JSON and contain the full translation table:

```json
{
  "session_id": "pii_69c2a72e_7aa396b9",
  "mappings": [
    {"original": "John Smith", "pseudonym": "Ashley Jackson", "type": "PERSON"},
    {"original": "Acme Corp", "pseudonym": "Lane, Randall and Hess", "type": "ORG"}
  ]
}
```

Keep these secure — they contain the original PII. Delete them when you no longer need round-trip restore.

## Transcript format

Transcript mode expects this format (common output from Gong, Otter.ai, Teams, etc.):

```
00:00:05 Speaker 1
This is what they said in the meeting.
00:00:58 John Smith
And here is another speaker's turn.
```

If no timestamps are found, it falls back to chunking by line count.

## Known limitations

- spaCy `en_core_web_sm` is fast but not the most accurate NER model. For higher accuracy, use `en_core_web_trf` (transformer-based, slower, requires PyTorch).
- First-name-only references ("Chris", "Drew") are caught but may collide across different people with the same first name in a session.
- Some product names and acronyms require adding to `NER_SKIP_TERMS` for your domain.
- The Faker seed is deterministic per entity type + original text. Different entity types with the same text will produce different pseudonyms.

## Attribution

Python port of [pii-pseudonymizer](https://github.com/jcfischer/pii-pseudonymizer) by [Jens-Christian Fischer](https://github.com/jcfischer). Original implementation in TypeScript/Bun under MIT license.

## License

MIT

---

## Northwoods Sentinel Labs

Part of the [Northwoods Sentinel Labs](https://northwoodssentinel.com) ecosystem — open-source tools for human-centered AI.

[Blog](https://northwoodssentinel.com) · [Substack](https://substack.com/@chewvala) · [GitHub](https://github.com/NorthwoodsSentinel)
