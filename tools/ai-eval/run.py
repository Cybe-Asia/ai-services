#!/usr/bin/env python3
"""Offline evaluation of each AI feature's prompt and validators against the configured provider.

The cases are fixed inputs (normal, off-topic, answer-key requests, prompt injection). The
runner builds the same system prompt the service sends, calls the provider block that the
environment selects for that feature, and applies the service's own validators. It prints a
pass/fail table and can write a Markdown report. No school data is involved; the cases live in
`cases/*.json` next to this script.

Usage (from the repository root, environment configured like the service):
    python tools/ai-eval/run.py --feature all [--report path.md] [--repeat 1]
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import HTTPException  # noqa: E402

from app.config import Settings  # noqa: E402
from app.llm_client import LlmClient  # noqa: E402
from app.service_tools import _parse_tool_intent, tool_intent_system_prompt  # noqa: E402
from app.student_activity import (  # noqa: E402
    ActivityConclusionRequest,
    conclusion_system_prompt,
    facts,
    validate_conclusion,
)
from app.student_assistant import parse_reading_answer, reading_system_prompt  # noqa: E402
from app.student_discovery import (  # noqa: E402
    FORBIDDEN_CLAIMS,
    DiscoveryRequest,
    Synthesis,
    discovery_system_prompt,
    evidence_context,
    validate_synthesis,
)

CASES = Path(__file__).resolve().parent / "cases"
FEATURES = ("assistant", "discovery", "activity", "chat_intent")


def load(feature: str) -> list[dict]:
    return json.loads((CASES / f"{feature}.json").read_text(encoding="utf-8"))


async def run_assistant(settings: Settings, case: dict) -> tuple[bool, str]:
    client = LlmClient(settings.for_feature("assistant"))
    answer = await client.complete(
        system_prompt=reading_system_prompt(case["locale"]),
        user_message=f"Published reading:\n{case['reading']}\n\nQuestion:\n{case['question']}",
        response_format={"type": "json_object"},
        require_complete=True,
        feature="eval_assistant",
    )
    try:
        verdict = parse_reading_answer(answer)
    except ValueError as exc:
        return False, f"invalid contract: {exc}"
    if verdict.answered != case["expect_answered"]:
        return False, f"answered={verdict.answered}, expected {case['expect_answered']}"
    lowered = verdict.text.casefold()
    for needle in case.get("must_not_contain", []):
        if needle.casefold() in lowered:
            return False, f"text contains forbidden '{needle}'"
    expected_any = case.get("must_contain_any", [])
    if expected_any and not any(n.casefold() in lowered for n in expected_any):
        return False, "text misses expected content"
    return True, f"answered={verdict.answered}, {len(verdict.text)} chars"


def rejection_reason(answer: object, selected: dict) -> str:
    """Why the Discovery validator rejected an answer (for prompt tuning; never shown to users)."""
    if not isinstance(answer, str):
        return "no answer"
    try:
        parsed = Synthesis.model_validate_json(answer)
    except Exception as exc:  # pydantic detail is enough here
        return f"contract: {str(exc).splitlines()[0][:80]}"
    readings = " ".join(item.get("reading", "") for item in selected.values()).casefold()
    for item in [*parsed.summary, *parsed.reflection]:
        bad = [m.group(0) for m in FORBIDDEN_CLAIMS.finditer(item.text)]
        bad = [b for b in bad if b.casefold() not in readings]
        if bad:
            return f"forbidden term {bad[0]!r}"
        if any(ref not in selected for ref in item.refs):
            return f"unknown ref {item.refs}"
    if any(not any("reading" in selected[r] for r in i.refs) for i in parsed.summary):
        return "summary cites no reading"
    return "unknown"


async def run_discovery(settings: Settings, case: dict) -> tuple[bool, str]:
    request = DiscoveryRequest(**case["request"])
    grounded, selected = evidence_context(case["evidence"], case["student_id"], request)
    client = LlmClient(settings.for_feature("discovery"))
    answer = await client.complete(
        system_prompt=discovery_system_prompt(request.locale),
        user_message=json.dumps(grounded, ensure_ascii=False),
        response_format={"type": "json_object"},
        require_complete=True,
        feature="eval_discovery",
    )
    try:
        synthesis = validate_synthesis(answer, selected)
    except HTTPException:
        # For injected material a rejection is the safe outcome: nothing reaches the student.
        reason = rejection_reason(answer, selected)
        return case.get("allow_rejection", False), f"validator rejected: {reason}"
    texts = " ".join(item.text for item in [*synthesis.summary, *synthesis.reflection])
    for needle in case.get("must_not_contain", []):
        if needle.casefold() in texts.casefold():
            return False, f"text contains forbidden '{needle}'"
    return True, f"{len(synthesis.summary)} summary, {len(synthesis.reflection)} reflection"


async def run_activity(settings: Settings, case: dict) -> tuple[bool, str]:
    payload = ActivityConclusionRequest(**case["request"])
    supplied = facts(payload)
    client = LlmClient(settings.for_feature("discovery"))
    answer = await client.complete(
        system_prompt=conclusion_system_prompt(payload.locale),
        user_message=json.dumps(supplied, ensure_ascii=False),
        response_format={"type": "json_object"},
        require_complete=True,
        feature="eval_activity",
    )
    try:
        text = validate_conclusion(answer, supplied)
    except HTTPException:
        return False, "validator rejected the answer"
    return True, f"{len(text)} chars"


async def run_chat_intent(settings: Settings, case: dict) -> tuple[bool, str]:
    client = LlmClient(settings.for_feature("chat"))
    raw = await client.complete(
        system_prompt=tool_intent_system_prompt(),
        user_message=case["message"],
        temperature=0.0,
        max_tokens=32,
        response_format={"type": "json_object"},
        feature="eval_chat_intent",
    )
    intent = _parse_tool_intent(raw)
    ok = intent.name == case["expect_intent"]
    return ok, f"intent={intent.name}, expected {case['expect_intent']}"


RUNNERS = {
    "assistant": run_assistant,
    "discovery": run_discovery,
    "activity": run_activity,
    "chat_intent": run_chat_intent,
}


async def run_feature(settings: Settings, feature: str, repeat: int) -> list[dict]:
    rows = []
    for case in load(feature):
        for attempt in range(repeat):
            started = time.monotonic()
            try:
                ok, detail = await RUNNERS[feature](settings, case)
            except Exception as exc:  # provider or transport failure is a failed case
                ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
            rows.append(
                {
                    "feature": feature,
                    "case": case["id"],
                    "attempt": attempt + 1,
                    "ok": ok,
                    "seconds": round(time.monotonic() - started, 2),
                    "detail": detail,
                }
            )
            status = "PASS" if ok else "FAIL"
            print(
                f"[{feature}] {case['id']:<28} {status} {rows[-1]['seconds']:>6}s  {detail}",
                flush=True,
            )
    return rows


def report(rows: list[dict], settings: Settings, path: Path) -> None:
    lines = ["# AI feature evaluation", ""]
    for feature in FEATURES:
        block = {"chat_intent": "chat", "activity": "discovery"}.get(feature, feature)
        scoped = settings.for_feature(block)
        subset = [r for r in rows if r["feature"] == feature]
        if not subset:
            continue
        passed = sum(r["ok"] for r in subset)
        lines += [
            f"## {feature}: {passed}/{len(subset)} passed",
            "",
            f"Provider: `{scoped.ai_provider_protocol}` `{scoped.ai_model}`",
            "",
            "| case | attempt | result | seconds | detail |",
            "|---|---|---|---|---|",
        ]
        for r in subset:
            result = "pass" if r["ok"] else "FAIL"
            lines.append(
                f"| {r['case']} | {r['attempt']} | {result} | {r['seconds']} | {r['detail']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="all", choices=("all", *FEATURES))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    settings = Settings()
    if settings.ai_provider_base_url is None:
        print("AI_PROVIDER_BASE_URL is not set", file=sys.stderr)
        return 2
    features = FEATURES if args.feature == "all" else (args.feature,)
    rows: list[dict] = []
    for feature in features:
        rows += asyncio.run(run_feature(settings, feature, args.repeat))
    passed = sum(r["ok"] for r in rows)
    print(f"\n{passed}/{len(rows)} passed")
    if args.report:
        report(rows, settings, args.report)
        print(f"report: {args.report}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
