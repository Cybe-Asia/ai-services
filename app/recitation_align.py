"""Word-level comparison of a recited verse with its expected text.

Diacritics-insensitive on purpose: the pre-check claims only that the right *word*
was heard (ADR 0009). Each expected word gets one of four statuses:

  ok       heard word is the expected word (similarity >= OK_THRESHOLD)
  unsure   close but not the same (UNSURE_THRESHOLD <= similarity < OK_THRESHOLD);
           the screen says "coba dengar lagi", nothing is called wrong
  wrong    a different word was heard
  missing  nothing aligned to this word

The thresholds are placeholders until student recordings calibrate them.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

OK_THRESHOLD = 0.8
UNSURE_THRESHOLD = 0.6

_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_ALIFS = re.compile(r"[ٱآأإ]")
_NON_ARABIC = re.compile(r"[^ء-ي\s]")


def normalize(text: str) -> list[str]:
    """Return the normalised word list of an Arabic string."""
    s = _MARKS.sub("", text)
    s = _ALIFS.sub("ا", s)
    s = (
        s.replace("ى", "ي")  # alif maqsura -> ya
        .replace("ؤ", "و")  # waw hamza -> waw
        .replace("ئ", "ي")  # ya hamza -> ya
        .replace("ء", "")  # bare hamza dropped
        .replace("ة", "ه")  # ta marbuta -> ha
    )
    s = _NON_ARABIC.sub(" ", s)
    return s.split()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def classify(sim: float, ok: float = OK_THRESHOLD, unsure: float = UNSURE_THRESHOLD) -> str:
    if sim >= ok:
        return "ok"
    if sim >= unsure:
        return "unsure"
    return "wrong"


@dataclass
class Verdict:
    index: int
    expected: str
    heard: str | None
    status: str
    similarity: float = 0.0
    heard_index: int | None = None  # position in the heard word list, when one was aligned


@dataclass
class Alignment:
    verdicts: list[Verdict]
    extras: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[int]:
        return [v.index for v in self.verdicts if v.status in ("wrong", "missing")]

    @property
    def unsure(self) -> list[int]:
        return [v.index for v in self.verdicts if v.status == "unsure"]


def align(
    expected: list[str],
    heard: list[str],
    ok_threshold: float = OK_THRESHOLD,
    unsure_threshold: float = UNSURE_THRESHOLD,
) -> Alignment:
    """Levenshtein alignment of two word lists with fuzzy substitution cost."""
    n, m = len(expected), len(heard)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = float(j)
        back[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = similarity(expected[i - 1], heard[j - 1])
            sub = dp[i - 1][j - 1] + (1.0 - sim)
            dele = dp[i - 1][j] + 1.0
            ins = dp[i][j - 1] + 1.0
            best = min(sub, dele, ins)
            dp[i][j] = best
            back[i][j] = "sub" if best == sub else ("del" if best == dele else "ins")

    verdicts: list[Verdict] = []
    extras: list[str] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "sub":
            e, h = expected[i - 1], heard[j - 1]
            sim = similarity(e, h)
            verdicts.append(
                Verdict(i - 1, e, h, classify(sim, ok_threshold, unsure_threshold), sim, j - 1)
            )
            i, j = i - 1, j - 1
        elif op == "del":
            verdicts.append(Verdict(i - 1, expected[i - 1], None, "missing", 0.0))
            i -= 1
        else:
            extras.append(heard[j - 1])
            j -= 1
    verdicts.reverse()
    extras.reverse()
    return Alignment(verdicts, extras)


HeardWord = tuple[str, float, float]  # text as transcribed, start and end in seconds


def heard_tokens(words: list[HeardWord]) -> tuple[list[str], list[tuple[float, float]]]:
    """Normalised tokens of the transcribed words with the timing of the word each came from."""
    tokens: list[str] = []
    timing: list[tuple[float, float]] = []
    for text, start, end in words:
        for token in normalize(text):
            tokens.append(token)
            timing.append((start, end))
    return tokens, timing


def assessment(expected_text: str, heard_text: str) -> dict:
    """The JSON shape the AI service returns to the Student screen, from a plain transcript."""
    return assessment_from_words(expected_text, [(heard_text, 0.0, 0.0)], timed=False)


def assessment_from_words(
    expected_text: str, heard_words: list[HeardWord], timed: bool = True
) -> dict:
    """As `assessment`, with where each aligned word sits in the recording (seconds)."""
    tokens, timing = heard_tokens(heard_words)
    al = align(normalize(expected_text), tokens)
    words = []
    for v in al.verdicts:
        word = {
            "index": v.index,
            "expected": v.expected,
            "heard": v.heard,
            "status": v.status,
            "similarity": round(v.similarity, 2),
        }
        if timed and v.heard_index is not None:
            start, end = timing[v.heard_index]
            word["heard_start"] = round(start, 2)
            word["heard_end"] = round(end, 2)
        words.append(word)
    return {"words": words, "extras": al.extras, "flagged": al.flagged, "unsure": al.unsure}
