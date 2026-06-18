# AGENTS.md

Read `/Users/arief/Developer/cybe/AGENTS.md` first, then `/Users/arief/Developer/cybe/digital schools/AGENTS.md`.

This is the Digital Schools AI service workspace.

## Service Notes

- Treat AI prompts, retrieved context, student records, admissions data, payments, and generated outputs as sensitive by default.
- Do not log raw prompts, bearer tokens, student names, grades, payment details, document identifiers, or model responses that may contain personal data.
- The AI service must not query Neo4j directly for sensitive operational data. Call safe, role-aware APIs exposed by owning Rust services instead.
- Keep the LLM provider swappable through OpenAI-compatible HTTP settings.
- Prefer deterministic tool/service results for facts and numbers. Use the model to explain or draft, not as the source of truth.
- Public/marketing answers must be grounded in approved content or clearly say that official source data is missing.
