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

## Boundaries

- `ai-services` may call `sis-services`, `admission-services`, and other owning services through explicit APIs.
- It must not bypass ownership boundaries with direct graph/database queries for sensitive data.
- Real student-grade answers must come from SIS APIs, not from model memory.
- Prompts and responses should be redacted or summarized in logs.
