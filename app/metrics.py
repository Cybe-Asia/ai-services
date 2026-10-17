"""In-process counters for the AI features.

Only feature names, outcomes, durations and token counts are kept; never prompt text,
model output, identities or request bodies. The registry is process-local (one pod).
"""

import re
import threading
from typing import Optional

LATENCY_BUCKETS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0)
_LABEL = re.compile(r"[^a-z0-9_]")


def _label(value: str) -> str:
    cleaned = _LABEL.sub("_", str(value).casefold())[:40]
    return cleaned or "unknown"


class Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._calls: dict[tuple[str, str], int] = {}
            self._latency: dict[str, list[int]] = {}
            self._latency_sum: dict[str, float] = {}
            self._tokens: dict[tuple[str, str], int] = {}
            self._outcomes: dict[tuple[str, str], int] = {}

    def observe_call(
        self, feature: str, outcome: str, seconds: float, usage: Optional[dict] = None
    ) -> None:
        """One provider call: its outcome, wall time and (when reported) token usage."""
        feature, outcome = _label(feature), _label(outcome)
        with self._lock:
            self._calls[(feature, outcome)] = self._calls.get((feature, outcome), 0) + 1
            buckets = self._latency.setdefault(feature, [0] * (len(LATENCY_BUCKETS) + 1))
            for index, bound in enumerate(LATENCY_BUCKETS):
                if seconds <= bound:
                    buckets[index] += 1
                    break
            else:
                buckets[-1] += 1
            self._latency_sum[feature] = self._latency_sum.get(feature, 0.0) + max(seconds, 0.0)
            for kind in ("input", "output"):
                count = (usage or {}).get(kind)
                if isinstance(count, int) and count > 0:
                    self._tokens[(feature, kind)] = self._tokens.get((feature, kind), 0) + count

    def count(self, feature: str, outcome: str) -> None:
        """A feature-level result after the provider answered (ok, validation_failed, ...)."""
        feature, outcome = _label(feature), _label(outcome)
        with self._lock:
            self._outcomes[(feature, outcome)] = self._outcomes.get((feature, outcome), 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": dict(self._calls),
                "latency": {k: list(v) for k, v in self._latency.items()},
                "latency_sum": dict(self._latency_sum),
                "tokens": dict(self._tokens),
                "outcomes": dict(self._outcomes),
            }

    def render(self) -> str:
        """Prometheus text exposition (version 0.0.4)."""
        data = self.snapshot()
        lines = [
            "# HELP ai_feature_calls_total Provider calls per feature and call outcome.",
            "# TYPE ai_feature_calls_total counter",
        ]
        for (feature, outcome), count in sorted(data["calls"].items()):
            lines.append(
                f'ai_feature_calls_total{{feature="{feature}",outcome="{outcome}"}} {count}'
            )
        lines += [
            "# HELP ai_feature_call_seconds Wall time of provider calls per feature.",
            "# TYPE ai_feature_call_seconds histogram",
        ]
        for feature, buckets in sorted(data["latency"].items()):
            cumulative = 0
            for index, bound in enumerate(LATENCY_BUCKETS):
                cumulative += buckets[index]
                lines.append(
                    f'ai_feature_call_seconds_bucket{{feature="{feature}",le="{bound:g}"}} '
                    f"{cumulative}"
                )
            cumulative += buckets[-1]
            lines.append(
                f'ai_feature_call_seconds_bucket{{feature="{feature}",le="+Inf"}} {cumulative}'
            )
            total = data["latency_sum"][feature]
            lines.append(f'ai_feature_call_seconds_sum{{feature="{feature}"}} {total:.3f}')
            lines.append(f'ai_feature_call_seconds_count{{feature="{feature}"}} {cumulative}')
        lines += [
            "# HELP ai_feature_tokens_total Tokens reported by the provider per feature.",
            "# TYPE ai_feature_tokens_total counter",
        ]
        for (feature, kind), count in sorted(data["tokens"].items()):
            lines.append(f'ai_feature_tokens_total{{feature="{feature}",kind="{kind}"}} {count}')
        lines += [
            "# HELP ai_feature_outcomes_total Feature results after validation per feature.",
            "# TYPE ai_feature_outcomes_total counter",
        ]
        for (feature, outcome), count in sorted(data["outcomes"].items()):
            lines.append(
                f'ai_feature_outcomes_total{{feature="{feature}",outcome="{outcome}"}} {count}'
            )
        return "\n".join(lines) + "\n"


registry = Registry()


def usage_from_body(body: dict) -> Optional[dict]:
    """Normalise OpenAI, Anthropic and Ollama usage fields to {input, output} token counts."""
    usage = body.get("usage")
    if isinstance(usage, dict):
        counts = {
            "input": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output": usage.get("completion_tokens", usage.get("output_tokens")),
        }
    elif "eval_count" in body or "prompt_eval_count" in body:
        counts = {"input": body.get("prompt_eval_count"), "output": body.get("eval_count")}
    else:
        return None
    return {k: v for k, v in counts.items() if isinstance(v, int) and v >= 0} or None
