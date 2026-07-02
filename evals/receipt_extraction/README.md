# Receipt Extraction Eval

First PoC from the AI enhancement plan: measure how accurately a vision
LLM (Qwen2.5-VL) extracts `{amount, currency, transfer_date, sender_name,
sender_bank, reference}` from Indonesian transfer receipts, to decide
whether it can pre-fill the finance review queue.

This is an offline eval harness. Nothing here runs in the service; the
accuracy number it produces decides whether the feature gets built.

## Layout

```text
extraction.py   Vision-LLM client (OpenAI-compatible + Ollama native) + prompt + JSON schema
scoring.py      Field normalization (Rp formats, Indonesian dates, bank aliases) + scoring
run_eval.py     CLI runner: manifest in, per-field accuracy + JSON report out
synth.py        Synthetic receipt generator (fake data, harness smoke tests only)
data/synthetic/ Generated fake receipts + manifest (safe to commit; no real PII)
data/staging/   Real staging receipts + manifest (NEVER commit; gitignored)
```

## Running

```bash
# one-time: Pillow is only needed to generate synthetic receipts
.venv/bin/pip install Pillow

# generate the synthetic smoke set
.venv/bin/python -m evals.receipt_extraction.synth \
    --out evals/receipt_extraction/data/synthetic

# run against a local Ollama serving qwen2.5vl
ollama pull qwen2.5vl:7b
.venv/bin/python -m evals.receipt_extraction.run_eval \
    --manifest evals/receipt_extraction/data/synthetic/manifest.jsonl \
    --base-url http://localhost:11434 \
    --model qwen2.5vl:7b \
    --out /tmp/receipt-eval-synthetic.json
```

`--api auto` detects Ollama from the URL and uses its native `/api/chat`
with JSON-schema constrained decoding (`format`). Any other URL uses
OpenAI-compatible `/chat/completions` with `response_format: json_object`
— so the same command works against the CYBE AI Gateway once it serves a
vision model.

## The real eval: staging receipts

Synthetic numbers are an upper bound on clean cases only. The decision
number needs 30–50 real payment proofs from staging with hand-checked
ground truth:

1. Export payment-proof images from staging (marketing-assist uploads).
2. For each, record the matching invoice/truth values in
   `data/staging/manifest.jsonl` (same format as the synthetic one).
3. Run `run_eval.py` with `--manifest data/staging/manifest.jsonl`.

Rules for staging data (children's/parents' data):

- Keep `data/staging/` out of git (gitignored here).
- Do not upload real receipts to any external API — self-hosted
  endpoints only.
- Delete the local copy when the eval round is done.

## Reading the results

- `amount` accuracy is the gate. A wrong amount silently corrupts the
  invoice comparison; anything under ~98% on `amount` means the feature
  must surface a confidence flag or drop pre-fill for that field.
- `sender_name`/`sender_bank` misses are cheap (finance sees the image
  next to the form), `reference` misses are annoying, `amount`/`date`
  misses are dangerous.
- `all-fields exact` is the headline number for the go/no-go decision in
  the AI enhancement plan.
