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
