"""Receipt-extraction eval runner.

Usage (from the ai-services repo root):

    .venv/bin/python -m evals.receipt_extraction.run_eval \
        --manifest evals/receipt_extraction/data/synthetic/manifest.jsonl \
        --base-url http://localhost:11434 \
        --model qwen2.5vl:7b \
        --out /tmp/receipt-eval-results.json

The manifest is JSONL; each line:
    {"image": "images/receipt_01.png", "truth": {"amount": 1500000, ...}}
Image paths are relative to the manifest file. Truth keys: amount (int),
currency, transfer_date (YYYY-MM-DD), sender_name, sender_bank, reference.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from evals.receipt_extraction.extraction import ReceiptExtractor
from evals.receipt_extraction.scoring import FIELDS, score_case, summarize


def load_manifest(manifest_path: Path) -> list:
    cases = []
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        entry = json.loads(line)
        image = manifest_path.parent / entry["image"]
        if not image.is_file():
            raise FileNotFoundError(f"manifest line {line_number}: image not found: {image}")
        cases.append((entry["image"], image, entry["truth"]))
    return cases


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).resolve()
    cases = load_manifest(manifest_path)
    if not cases:
        print("manifest is empty", file=sys.stderr)
        return 1

    extractor = ReceiptExtractor(
        base_url=args.base_url,
        model=args.model,
        api=args.api,
        api_key=args.api_key,
        timeout_seconds=args.timeout,
    )

    results = []
    for index, (name, image, truth) in enumerate(cases, start=1):
        started = time.monotonic()
        error = None
        prediction = None
        try:
            prediction = extractor.extract(image)
        except Exception as exc:  # noqa: BLE001 - eval must survive per-case failures
            error = f"{type(exc).__name__}: {exc}"
        latency = time.monotonic() - started
        result = score_case(name, truth, prediction, latency, error)
        results.append(result)

        status = "OK " if result.all_correct else "MISS"
        wrong = [f for f in FIELDS if not result.field_correct.get(f)]
        detail = error or (", ".join(wrong) if wrong else "")
        print(f"[{index}/{len(cases)}] {status} {name} ({latency:.1f}s) {detail}")

    summary = summarize(results)
    print()
    print(f"cases: {summary['cases']}  parse failures: {summary['parse_failures']}")
    print(f"all-fields exact: {summary['all_fields_correct']:.0%}")
    for field_name, accuracy in summary["per_field_accuracy"].items():
        print(f"  {field_name:<14} {accuracy:.0%}")
    print(f"mean latency: {summary['mean_latency_seconds']:.1f}s")

    if args.out:
        report = {
            "model": args.model,
            "base_url": args.base_url,
            "manifest": str(manifest_path),
            "summary": summary,
            "cases": [
                {
                    "image": r.image,
                    "truth": r.truth,
                    "prediction": r.prediction,
                    "field_correct": r.field_correct,
                    "latency_seconds": round(r.latency_seconds, 2),
                    "error": r.error,
                }
                for r in results
            ],
        }
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"report written to {args.out}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the receipt-extraction eval")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--base-url", required=True, help="Provider base URL")
    parser.add_argument("--model", default="qwen2.5vl:7b")
    parser.add_argument(
        "--api",
        choices=["auto", "ollama", "openai"],
        default="auto",
        help="Provider API flavor (auto-detects Ollama from the URL)",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-case timeout seconds")
    parser.add_argument("--out", default=None, help="Write full JSON report here")
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
