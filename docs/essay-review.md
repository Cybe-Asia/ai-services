# Teacher essay review suggestions

`POST /api/ai/v1/teacher/essay-review` — AI suggestions for a teacher reviewing one submitted
attempt (Digital Schools ADR 0003). The caller's own Learning session reads the attempt; this
service adds no authority, saves nothing, and never shows anything to students. The teacher
edits the suggestion and saves with Learning's existing `review` command.

## Request

Headers: `Authorization: Bearer <teacher Learning token>`, `Content-Type: application/json`.
Body (≤ 2 KB, unknown keys refused):

```json
{
  "class_id": "SEC-…",
  "course_id": "…",
  "assessment_id": "…",
  "attempt_id": "…",
  "locale": "id"
}
```

`locale` (`en` | `id`, default `id`) selects the language of the prompt, rubric and criteria
passed to the model and of the feedback draft.

## What happens

1. `GET …/assessments/courses/{course}/items/{assessment}` and
   `GET …/items/{assessment}/attempts/{attempt}` are read from Learning with the caller's
   token. Learning enforces that the caller teaches the class. The assessment must belong to
   `class_id`; the attempt must be `submitted` or `reviewed`.
2. A Student projection (no `rubric`/`criteria` keys on essay questions) is refused with 403:
   only a Teacher projection carries criteria.
3. For each essay question that has `criteria` and a non-empty text answer, one model call
   through the `ESSAY_*` provider block (`Settings.for_feature("essay")`), at most four in
   parallel. System message: question prompt, rubric, model answer, criteria (id and
   description) and the output contract; user message: the student's answer only. No student
   name or id, class, score or other answers are sent.
4. The model's JSON is validated in code (any failure ⇒ no suggestion for that answer):
   - every criterion id exactly once, no unknown ids;
   - `met`/`partial` need at least one evidence quote; every quote must occur in the answer
     (whitespace differences tolerated, the returned quote is the exact slice of the answer);
   - `reason` (≤ 600 chars) and `feedback` (≤ 1200 chars) pass a claims filter: nothing about
     the student's ability, traits, diagnoses, effort or other students, no score/grade
     numbers or percentages;
   - only a finished completion is accepted.
5. The score is computed in code: `met` = 100 %, `partial` = 50 %, `not_met`/`unclear` = 0 %
   of the criterion's points, half shares rounded half up to the point unit (0.01).
6. The attempt is read again after inference; a changed `revision` or `state` is a 409.

## Response

```json
{
  "contractVersion": 1,
  "attempt_id": "…",
  "revision": 3,
  "locale": "id",
  "questions": [
    {
      "question_id": "…",
      "prompt_version": "essay-review/1",
      "model": "gpt-6-luna",
      "maximum": 10,
      "suggested_score": 8,
      "needs_teacher": false,
      "flags": [],
      "criteria": [
        {"id": "bahan", "description": "…", "points": 4, "verdict": "met",
         "evidence": ["cahaya matahari, air dari akar, dan karbon dioksida"], "reason": "…"}
      ],
      "feedback": "…"
    }
  ]
}
```

One entry per essay question with criteria, in attempt order. Questions without criteria,
and non-essay questions, are not listed. `flags` ⊆ `needs_teacher`, `validation_failed`,
`provider_unavailable`, `empty_answer`; when the suggestion is missing, `suggested_score`,
`feedback` and every `verdict` are `null`. `needs_teacher` is also set when any criterion is
`unclear`.

## Errors

| Status | Meaning |
|---|---|
| 401 | No or malformed bearer token |
| 403 | Learning refused the caller, or the projection is not a Teacher projection |
| 404 | `ESSAY_*` block not configured (feature off); assessment not in `class_id`; attempt unknown |
| 409 | Attempt not `submitted`/`reviewed`, or it changed during inference |
| 413 / 422 | Body too large / invalid selection |
| 503 | Learning unavailable, malformed projection, or every model call failed; also the 55 s overall limit |

## Internal endpoint for Learning (ADR 0010)

`POST /api/ai/v1/internal/essay-review`, bearer = `ESSAY_INTERNAL_TOKEN` (shared with
learning-api, which calls it in the background when a student submits an attempt). Body:

```json
{"locale": "id", "questions": [{"question_id": "…", "prompt": "…", "rubric": "…",
  "model_answer": "…", "points": 10,
  "criteria": [{"id": "bahan", "description": "…", "points": 4}], "answer": "…"}]}
```

Same prompt, validation and scoring as the teacher endpoint; the response has the same
`questions[]` shape (no `attempt_id`/`revision`). Nothing is read from Learning and nothing
is kept. 401 without the token, 404 when the `ESSAY_*` block is unset, 422 for malformed
tasks (criteria must add up to the points), 503 when every model call failed.

## Configuration

`ESSAY_PROVIDER_PROTOCOL` (`openai` default | `anthropic` | `openai_compatible`),
`ESSAY_PROVIDER_BASE_URL`, `ESSAY_PROVIDER_API_KEY`, `ESSAY_MODEL`, `ESSAY_MAX_TOKENS`
(default 1200), `ESSAY_REASONING_EFFORT` (default `low`), `ESSAY_TIMEOUT_SECONDS`
(default 30), `ESSAY_INTERNAL_TOKEN` (internal endpoint only). All three of URL, key and model must be set; otherwise the endpoint answers 404
and `GET /api/v1/ai-service/metadata` reports `essayReviewEnabled: false`, which the Teacher
web uses to hide the button. There is deliberately no fallback to the main provider.

## Observability and evaluation

Metrics under feature `essay`: `ai_feature_calls_total`, `ai_feature_call_seconds`,
`ai_feature_tokens_total` (provider calls) and `ai_feature_outcomes_total` with outcomes
`ok`, `validation_failed`, `provider_unavailable`, `empty_answer`. Prompts, answers and model
output are never logged.

`python tools/ai-eval/run.py --feature essay --report docs/evidence/ai-eval-<date>-essay.md`
runs the fixed cases in `tools/ai-eval/cases/essay.json` through the same prompt and
validators. Re-run on every prompt (`prompt_version`) or model change; reports live in
`docs/evidence/`.
