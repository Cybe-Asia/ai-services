# Benchmark results

Model: `qwen2.5vl:7b` (Ollama, JSON-schema constrained decoding, temperature 0).
Set: 30 synthetic cases — 15 clean across 12 bank/e-wallet apps (incl. 3 dark
mode), 12 degraded (WhatsApp double-compression, low-res JPEG, blur, skewed
photo), 2 admin-fee traps, 3 non-receipt distractors (incl. a chat screenshot
that mentions an amount). Hardware: M4 MacBook, ~15-16s/case.

| Field | v1 prompt | v2 prompt (bank-branding hint) |
|---|---|---|
| is_receipt | 100% | 100% |
| amount | 100% | 100% |
| currency | 100% | 100% |
| transfer_date | 100% | 100% |
| sender_name | 100% | 100% |
| sender_bank | 23% (all honest nulls) | 100% |
| reference | 97% | 97% |
| all-fields exact | 23% | **97%** |

Findings worth keeping:

- The v1 sender_bank misses were all abstentions (null), never wrong guesses —
  the "never guess" prompt rule works. The fix was telling the model that app
  branding identifies the bank, which is true on real receipts.
- The one persistent miss is a digit misread in a reference number on the
  low-res case (556677 -> 558877). Amounts survived every degradation, but
  reference digits from degraded images should not be trusted for exact-match
  logic (e.g. duplicate detection) without a resolution check.
- The chat-screenshot trap (mentions "Rp1.500.000" in a bubble) was correctly
  rejected via is_receipt=false with all fields null.

Caveat: synthetic receipts are rendered, not photographed — real WhatsApp
photos of paper slips will be worse. This number is the interim gate while
real receipts are unavailable; post-launch, every finance correction becomes
a labeled real case for re-running this eval.
