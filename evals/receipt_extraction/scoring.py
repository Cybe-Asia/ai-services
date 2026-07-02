"""Field normalization and scoring for receipt-extraction evals.

Ground truth and predictions are compared field by field after
normalization, so formatting differences ("Rp1.500.000,00" vs 1500000,
"01 Juli 2026" vs "2026-07-01") never count as extraction errors.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FIELDS = ("amount", "currency", "transfer_date", "sender_name", "sender_bank", "reference")

_INDONESIAN_MONTHS = {
    "januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6,
    "juli": 7, "agustus": 8, "september": 9, "oktober": 10, "november": 11,
    "desember": 12, "january": 1, "february": 2, "march": 3, "may": 5,
    "june": 6, "july": 7, "august": 8, "october": 10, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "agu": 8,
    "aug": 8, "sep": 9, "okt": 10, "oct": 10, "nov": 11, "des": 12, "dec": 12,
}

_BANK_ALIASES = {
    "bank central asia": "bca",
    "bank mandiri": "mandiri",
    "livin by mandiri": "mandiri",
    "livin": "mandiri",
    "bank negara indonesia": "bni",
    "bank rakyat indonesia": "bri",
    "brimo": "bri",
    "seabank indonesia": "seabank",
    "bank jago": "jago",
    "cimb niaga": "cimb",
}


def normalize_amount(value: object) -> int | None:
    """Parse an amount into integer rupiah.

    Handles Indonesian formatting (dots as thousands, comma decimals),
    Western formatting, and currency prefixes. Decimal fractions are
    dropped (IDR has no circulating subunit).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).strip().lower()
    text = re.sub(r"(rp|idr)\.?\s*", "", text)
    text = text.replace(" ", "")
    if not text:
        return None

    # Comma followed by exactly 2 digits at the end = decimal part -> drop it.
    text = re.sub(r",\d{2}$", "", text)
    # A dot followed by exactly 2 digits at the end is also decimal ("1500000.00").
    text = re.sub(r"\.\d{2}$", "", text)
    # Remaining dots/commas are thousands separators.
    text = text.replace(".", "").replace(",", "")

    if not re.fullmatch(r"\d+", text):
        return None
    return int(text)


def normalize_date(value: object) -> str | None:
    """Normalize a date to ISO YYYY-MM-DD. Returns None if unparseable."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    # "01 juli 2026", "1 jul 2026 10:23"
    match = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", text)
    if match:
        month = _INDONESIAN_MONTHS.get(match.group(2))
        if month:
            return f"{match.group(3)}-{month:02d}-{int(match.group(1)):02d}"

    # "01/07/2026" or "01-07-2026" -> assume day first (Indonesian convention)
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", text)
    if match:
        return f"{match.group(3)}-{int(match.group(2)):02d}-{int(match.group(1)):02d}"

    return None


def normalize_name(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^a-z0-9 ]", "", str(value).casefold())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_bank(value: object) -> str | None:
    text = normalize_name(value)
    if text is None:
        return None
    if text in _BANK_ALIASES:
        return _BANK_ALIASES[text]
    text = re.sub(r"^(pt|bank)\s+", "", text)
    return _BANK_ALIASES.get(text, text)


def normalize_reference(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    return text or None


def _match_amount(truth: object, pred: object) -> bool:
    truth_amount, pred_amount = normalize_amount(truth), normalize_amount(pred)
    return truth_amount is not None and truth_amount == pred_amount


def _match_date(truth: object, pred: object) -> bool:
    normalized = normalize_date(truth)
    return normalized is not None and normalized == normalize_date(pred)


def _match_name(truth: object, pred: object) -> bool:
    truth_name, pred_name = normalize_name(truth), normalize_name(pred)
    if truth_name is None or pred_name is None:
        return truth_name == pred_name
    return truth_name == pred_name or truth_name in pred_name or pred_name in truth_name


def _match_bank(truth: object, pred: object) -> bool:
    truth_bank, pred_bank = normalize_bank(truth), normalize_bank(pred)
    if truth_bank is None or pred_bank is None:
        return truth_bank == pred_bank
    return truth_bank == pred_bank or truth_bank in pred_bank or pred_bank in truth_bank


def _match_currency(truth: object, pred: object) -> bool:
    normalize = lambda v: str(v).strip().casefold().replace("rp", "idr") if v else None  # noqa: E731
    return normalize(truth) == normalize(pred)


def _match_reference(truth: object, pred: object) -> bool:
    return normalize_reference(truth) == normalize_reference(pred)


_MATCHERS = {
    "amount": _match_amount,
    "currency": _match_currency,
    "transfer_date": _match_date,
    "sender_name": _match_name,
    "sender_bank": _match_bank,
    "reference": _match_reference,
}


@dataclass
class CaseResult:
    image: str
    truth: dict
    prediction: dict | None
    field_correct: dict = field(default_factory=dict)
    latency_seconds: float = 0.0
    error: str | None = None

    @property
    def all_correct(self) -> bool:
        return bool(self.field_correct) and all(self.field_correct.values())


def score_case(
    image: str,
    truth: dict,
    prediction: dict | None,
    latency_seconds: float,
    error: str | None = None,
) -> CaseResult:
    result = CaseResult(
        image=image,
        truth=truth,
        prediction=prediction,
        latency_seconds=latency_seconds,
        error=error,
    )
    for name in FIELDS:
        if prediction is None:
            result.field_correct[name] = False
        else:
            result.field_correct[name] = _MATCHERS[name](truth.get(name), prediction.get(name))
    return result


def summarize(results: list) -> dict:
    total = len(results)
    if total == 0:
        return {"cases": 0}
    per_field = {
        name: sum(1 for r in results if r.field_correct.get(name)) / total for name in FIELDS
    }
    return {
        "cases": total,
        "parse_failures": sum(1 for r in results if r.prediction is None),
        "per_field_accuracy": per_field,
        "all_fields_correct": sum(1 for r in results if r.all_correct) / total,
        "amount_accuracy": per_field["amount"],
        "mean_latency_seconds": sum(r.latency_seconds for r in results) / total,
    }
