# ai-services

Digital Schools AI gateway service.

This service owns the product-facing AI API for Digital Schools. It performs prompt policy checks, routes safe tool calls to the owning Rust services, and calls an OpenAI-compatible model provider for generation.

## Local Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8082
```

Health:

```bash
curl http://127.0.0.1:8082/api/v1/ai-service/health
```

Chat smoke:

```bash
curl -s http://127.0.0.1:8082/api/ai/v1/chat \
  -H 'content-type: application/json' \
  -d '{"message":"buatkan ide marketing ppdb","actorRole":"public","locale":"id"}' | jq
```

Useful local/provider settings:

- `AI_PROVIDER_BASE_URL`: OpenAI-compatible provider URL, for example Ollama `/v1`.
- `AI_MODEL`: provider model name.
- `AI_MAX_TOKENS`: response cap sent to the provider.
- `REQUEST_TIMEOUT_SECONDS`: provider request timeout.
- `ADMISSION_SERVICE_URL`: admission-service base URL for owner/admin aggregate tools.
- `PAYMENT_SERVICE_URL`: payment-service base URL for owner/admin aggregate tools.

## Boundaries

- `ai-services` may call `sis-services`, `admission-services`, and other owning services through explicit APIs.
- It must not bypass ownership boundaries with direct graph/database queries for sensitive data.
- Real student-grade answers must come from SIS APIs, not from model memory.
- Admission and payment facts must come from role-aware service APIs, not from model memory.
- Prompts and responses should be redacted or summarized in logs.

## Tool Routing

Owner/admin factual prompts are resolved through safe tools. The service first handles known
patterns deterministically, then uses the configured Qwen/OpenAI-compatible provider as an
intent classifier for paraphrased questions. The model returns an approved tool intent only; the
actual totals still come from `admission-service` or `payment-service`.

## Private document analysis

`POST /api/ai/v1/internal/document-analysis` accepts raw PNG, JPEG, or PDF
bytes from admission-service only. It requires `DOCUMENT_ANALYSIS_ENABLED=true`,
a Bearer token matching `DOCUMENT_ANALYSIS_INTERNAL_TOKEN`, and
`x-expected-document-type: birth_certificate`. Configure
`DOCUMENT_ANALYSIS_MODEL=qwen2.5vl:7b`; no external provider fallback is used.

The endpoint validates signatures, image integrity, PDF encryption/page count,
renders at a bounded DPI, invokes the configured OpenAI-compatible gateway at
temperature zero, and validates schema `1.0`. Do not log request bodies, model
responses, extracted fields, or raw prompts.

## Per-feature providers, metrics and evaluation

Every feature calls the same `LlmClient`; which provider answers is configuration only.
`AI_PROVIDER_*` is the main provider (the on-prem gateway). Three features can be moved to
their own provider by setting a complete block; an incomplete block leaves that feature on
the main provider and changes nothing else:

| Block | Features | Variables |
|---|---|---|
| `DISCOVERY_*` | Student Discovery reflection, Activity Arena conclusion | `DISCOVERY_PROVIDER_PROTOCOL`, `DISCOVERY_PROVIDER_BASE_URL`, `DISCOVERY_PROVIDER_API_KEY`, `DISCOVERY_MODEL`, `DISCOVERY_MAX_TOKENS`, `DISCOVERY_REASONING_EFFORT`, `DISCOVERY_TIMEOUT_SECONDS` |
| `ASSISTANT_*` | Kibo: explanation of one selected published reading | same names with the `ASSISTANT_` prefix |
| `CHAT_*` | Admin assist: intent classification and drafting | same names with the `CHAT_` prefix |
| `ESSAY_*` | Teacher essay-review suggestions (`POST /api/ai/v1/teacher/essay-review`, see `docs/essay-review.md`) | same names with the `ESSAY_` prefix; **no fallback**: an incomplete block turns the feature off (404, `essayReviewEnabled: false` in metadata) because it sends student-written text |

Protocols: `openai` (Chat Completions with `max_completion_tokens` and `reasoning_effort`),
`anthropic` (Messages API) and `openai_compatible` (the gateway or a local model; JSON mode
is sent as `response_format: json_object`, qwen3 gets `think: false`). Moving a feature means
changing those four values and restarting; the model still has to pass the feature's eval set
(below), because JSON mode and reasoning controls differ between models.

Kibo's explanation returns `{"answered": bool, "text": string}`: when the selected reading
does not cover the question, or the question is not about the reading, the student gets a
fixed "choose another reading" reply instead of a model answer.

`CHAT_OPEN_DRAFTING` (default `false`): when no school tool matches an owner/admin/teacher
question, Admin assist answers with the tool catalogue instead of a free model draft. Public
chat keeps drafting. Set `true` to restore open drafting for staff.

`GET /api/v1/ai-service/metrics` exposes Prometheus text: `ai_feature_calls_total` (per
feature and call outcome: ok, empty, timeout, error), `ai_feature_call_seconds` (histogram),
`ai_feature_tokens_total` (when the provider reports usage) and `ai_feature_outcomes_total`
(after validation: ok, validation_failed, not_answered). Counts only; never prompt or output
text.

`tools/ai-eval/run.py` runs each feature's prompt and validators against the provider in the
environment with fixed cases (normal, off-topic, answer-key requests, prompt injection; for
`essay` also graded answers with the verdicts each criterion may receive):

```bash
AI_PROVIDER_BASE_URL=http://127.0.0.1:11434/v1 AI_MODEL=qwen3:4b \
  python tools/ai-eval/run.py --feature all --report docs/evidence/ai-eval-<date>.md
```

Run it after every model or prompt change; the report lists pass/fail per case with latency
and token counts.
