# Teacher question drafts

`POST /api/ai/v1/teacher/question-drafts` drafts questions for a teacher's subject bank from a module's own chapter
readings. It is the follow-up ADR 0011 §9 deferred. The design is in
`digital-school/docs/learning/ai-question-drafting-spec.md`.

- **Caller:** the Teacher web (learning-web gateway `/api/teacher/learning/question-drafts`) with the teacher's own
  bearer token. The service reads with that token and adds no authority:
  - the module: `GET /api/v1/learning/curriculum/courses/{id}`, published syllabus only;
  - the teacher's published bank for the module's subject, for duplicates.

  No student data is read or sent.
- **Feature switch:** the `DRAFTS_*` provider block (`drafts_provider_base_url`, `_api_key`, `_model`). Unset answers
  404, and the Teacher web hides the action.
- **Request** (8 KiB at most):
  `{course_id, chapter_ids[1..10], lesson_ids[≤50]?, count 1..10, types "auto"|[types], difficulty
  "auto"|"easy"|"medium"|"hard", note ≤500, locale "en"|"id"}`.
- **Material:**
  - only the text blocks of the chosen lessons (heading, paragraph, formula, note, example, list and table, in both
    languages);
  - at most 40,000 characters, cut in syllabus order (flag `material_truncated`);
  - a chapter set with no text reading answers 409 before the model is called.
- **Response:** `{contract_version: 1, prompt_version: "question-drafts/1", model, course_id, course_revision,
  subject, requested, drafts[], discarded{invalid, no_quote, duplicate, type}, flags[]}`.
  - Each draft is `{question, metadata, evidence{text, chapter_id, lesson_id}, type_reason}`.
  - `question` is Learning's bank wire shape (id `ai-…`, options `o1…`, matching right `r1…`, essay criteria `c1…`).
  - `metadata` is `{grade: null, topics, difficulty, source{course_id, course_revision, chapter_id, lesson_id}}`.
- **What is kept:** a draft survives only when:
  - it validates for its type:
    - mcq: 3–5 options, one answer;
    - multi: 3–5 options, at least one answer;
    - tf: true or false;
    - short: 1–5 accepted answers;
    - numeric: value ± tolerance;
    - matching: 3–6 pairs;
    - essay: a rubric and 1–4 criteria, points = their sum;
  - its type is one the teacher allowed;
  - its `evidence` quote is found in a chosen lesson's text (case and whitespace tolerant; it moves to the lesson it
    is actually in);
  - neither prompt matches the bank or another draft (the bank's fingerprint: case, punctuation and spacing ignored).
- **Failures:** provider unavailable or an unparseable answer → 503 (no heuristic: invented questions are worse than
  none). A second request from the same token while one runs → 429. The whole call stops after 55 s.
- **Metrics:** feature `drafts`, outcomes `ok`, `none_usable`, `provider_unavailable` and `validation_failed`.
  No prompt text.
- **Saving** is the Teacher web's: accepted drafts go through the bank's `author` command as drafts, and Learning
  validates them (including duplicates) as any hand-written question.
