# ai-services

FastAPI service for the Digital Schools AI assistant. It acts as a safe,
role-aware AI gateway for school admins and public users.

Read `AGENTS.md` first. This file adds service-local implementation notes.

---

## 1. Purpose And Boundaries

This service owns:

- AI chat endpoints and SSE streaming responses.
- Conversation/thread storage for the admin assistant.
- Deterministic tool routing to owning services.
- Report/export generation for AI-produced operational summaries.
- LLM-provider integration through OpenAI-compatible HTTP settings.

This service does not own admissions, payments, SIS, auth, or school records.
For factual data, call safe role-aware APIs from the owning services:

- `admission-service` for EOI, leads, applications, applicant students, and admissions state.
- `payment-service` for payment review queues, fee structures, and finance state.
- future SIS service APIs for enrolled student, attendance, grades, and timetable data.

Do not query Neo4j directly from this service for operational school data.

---

## 2. Sensitive Data Rules

Treat prompts, retrieved context, generated answers, exports, student records,
parent details, admissions data, and payments as sensitive.

- Do not log raw prompts, bearer tokens, student names, parent names, emails,
  payment details, document ids, or generated responses that may contain PII.
- Keep factual answers grounded in service tool results.
- Do not let the model invent counts, names, payment status, prices, grades, or dates.
- Public users must only receive public/approved answers.
- Admin users may query operational data only through authenticated downstream APIs.

Thread context is routing memory only. It may help infer the next safe tool, but
it is not a factual cache. Always fetch fresh facts from the owning service.

---

## 3. Architecture

Important files:

```text
app/main.py            FastAPI routes, SSE stream orchestration, thread persistence
app/service_tools.py   Deterministic school-data tools, routing, reports, exports
app/thread_store.py    Server-side AI thread store and compact routing context
app/llm_client.py      OpenAI-compatible LLM client
app/settings.py        Environment-driven service configuration
app/schemas.py         Request/response DTOs
tests/                Pytest coverage for routing, streaming, reports, and tools
```

Typical answer flow:

```text
HTTP request
  -> resolve actor/admin context
  -> hydrate server-side thread history/context
  -> deterministic tool routing for factual school data
  -> optional LLM drafting for non-factual copy
  -> persist exchange + compact routing context
```

Prefer deterministic tool results for school facts. Use the LLM for language,
marketing drafts, summarization, and explanation.

---

## 4. Code Quality And SOLID

Write AI-service code as clean, layered application code, not as prompt glue.

SOLID expectations:

- Single Responsibility: route handlers handle HTTP concerns; tool functions
  handle school-data retrieval; report builders handle artifact generation;
  LLM clients handle provider calls only.
- Open/Closed: add new tools, report formats, or providers by extending small
  functions/classes instead of rewriting the main chat flow.
- Liskov Substitution: provider clients and tool results should keep stable
  request/response contracts so Ollama, Qwen, Llama, or another
  OpenAI-compatible model can be swapped without changing business logic.
- Interface Segregation: keep DTOs, tool calls, thread storage, report
  generation, and LLM calls separated. Do not create one giant assistant
  service that knows every detail.
- Dependency Inversion: high-level chat orchestration should depend on
  `Settings`, typed schemas, and small helper interfaces; avoid hardcoded
  URLs, model names, or environment assumptions inside business logic.

Clean-code expectations:

- Prefer small named helpers over long branching blocks.
- Keep factual routing deterministic and covered by regression tests.
- Keep prompt/model behavior isolated from service-data ownership rules.
- Validate inputs at API boundaries and normalize dates in one place.
- Return explicit safe fallback answers when a tool is unavailable.
- Avoid hidden side effects, broad mutable globals, and unrelated refactors.
- Add tests for every new intent, follow-up pattern, date range, export type,
  or sensitive-data branch.

---

## 5. LLM Provider

The provider must remain swappable through OpenAI-compatible settings. Current
staging uses local Ollama/OpenAI-compatible routing, but code should not depend
on a specific model family.

When changing model behavior:

- Keep the model behind `LlmClient`.
- Keep factual data in tools, not prompts.
- Add regression tests for routing and follow-up context.
- Avoid model-specific response parsing unless guarded and tested.

---

## 6. Conversation Context

The admin assistant supports server-side threads. Thread messages may be pruned
for retention/size, so `thread_store` also keeps compact structured context:

- last successful tool
- recent successful tools
- last date range
- last payment status
- last lead/person query
- reportable flag

This context exists only to preserve follow-up intent such as `detailnya`,
`siapa`, or `convert ke pdf dong` after older messages are pruned.

Do not store authoritative facts in this context.

---

## 7. Local Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8090
```

Useful checks:

```bash
. .venv/bin/activate && python -m ruff check app tests
. .venv/bin/activate && pytest
```

Targeted tests while changing routing:

```bash
. .venv/bin/activate && pytest tests/test_main.py -k "context or eoi or payment or report"
```

---

## 8. Deployment Notes

Staging branch pushes build `ghcr.io/cybe-asia/ai-services:sha-<commit>`.
The workflow currently needs a `GITOPS_TOKEN` secret to auto-bump
`digital-school-gitops/lab/staging`. If that secret is missing, bump the
staging overlay manually with a clean GitOps checkout.

Validate GitOps changes with:

```bash
kustomize build lab/staging >/tmp/ds-gitops-staging-render.yaml
rg -n "image: ghcr.io/cybe-asia/ai-services" /tmp/ds-gitops-staging-render.yaml
```

Observe CYBE staging through `ssh cybe`; do not assume staging runs on
Hostinger.

---

## 9. Gotchas

- Streaming via SSE can look frozen if tool calls or model startup take too
  long before the first token. Emit status events early and keep tool calls
  deterministic.
- Browser console noise from password-manager extensions is not necessarily an
  app bug. Reproduce in a clean profile before chasing extension stack traces.
- Relative dates must be computed in `Asia/Jakarta`.
- English and Indonesian prompts should route to the same tools; the response
  language should follow the user's prompt/locale when possible.
- Exports should return downloadable artifacts without exposing internal tool
  names to admins unless debugging is explicitly enabled.
