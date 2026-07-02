from evals.receipt_extraction.extraction import parse_prediction
from evals.receipt_extraction.scoring import (
    normalize_amount,
    normalize_bank,
    normalize_date,
    score_case,
    summarize,
)


def test_normalize_amount_indonesian_formats():
    assert normalize_amount("Rp1.500.000,00") == 1500000
    assert normalize_amount("Rp 2.750.000") == 2750000
    assert normalize_amount("IDR 1,750,000.00") == 1750000
    assert normalize_amount("1500000") == 1500000
    assert normalize_amount(1500000) == 1500000
    assert normalize_amount(1500000.0) == 1500000
    assert normalize_amount("Rp250.000") == 250000
    assert normalize_amount(None) is None
    assert normalize_amount("tidak terbaca") is None


def test_normalize_date_formats():
    assert normalize_date("2026-06-28") == "2026-06-28"
    assert normalize_date("28 Juni 2026 09:41:22") == "2026-06-28"
    assert normalize_date("25/06/2026 14:07") == "2026-06-25"
    assert normalize_date("30 Jun 2026, 19:23") == "2026-06-30"
    assert normalize_date("01 Juli 2026") == "2026-07-01"
    assert normalize_date(None) is None


def test_normalize_bank_aliases():
    assert normalize_bank("Bank Central Asia") == "bca"
    assert normalize_bank("BCA") == "bca"
    assert normalize_bank("Livin' by Mandiri") == "mandiri"
    assert normalize_bank("SeaBank Indonesia") == "seabank"


def test_score_case_matches_equivalent_formats():
    truth = {
        "amount": 1500000,
        "currency": "IDR",
        "transfer_date": "2026-06-28",
        "sender_name": "Budi Santoso",
        "sender_bank": "BCA",
        "reference": "95031428796541",
    }
    prediction = {
        "amount": "Rp1.500.000,00",
        "currency": "Rp",
        "transfer_date": "28 Juni 2026",
        "sender_name": "BUDI SANTOSO",
        "sender_bank": "Bank Central Asia",
        "reference": "9503-1428-796541",
    }
    result = score_case("x.png", truth, prediction, latency_seconds=1.0)
    assert result.all_correct


def test_score_case_catches_magnitude_error():
    truth = {"amount": 1500000, "currency": "IDR", "transfer_date": "2026-06-28",
             "sender_name": "Budi", "sender_bank": "BCA", "reference": "1"}
    prediction = dict(truth, amount=150000000)
    result = score_case("x.png", truth, prediction, latency_seconds=1.0)
    assert not result.field_correct["amount"]
    assert not result.all_correct


def test_score_case_null_prediction_counts_as_miss():
    truth = {"amount": 1, "currency": "IDR", "transfer_date": "2026-01-01",
             "sender_name": "A", "sender_bank": "BCA", "reference": "1"}
    result = score_case("x.png", truth, None, latency_seconds=0.0, error="timeout")
    assert not result.all_correct
    summary = summarize([result])
    assert summary["parse_failures"] == 1
    assert summary["all_fields_correct"] == 0.0


def test_parse_prediction_tolerates_fences_and_prose():
    assert parse_prediction('{"amount": 1}') == {"amount": 1}
    assert parse_prediction('```json\n{"amount": 1}\n```') == {"amount": 1}
    assert parse_prediction('Here you go: {"amount": 1} hope that helps') == {"amount": 1}
    assert parse_prediction("no json here") is None
    assert parse_prediction(None) is None
    assert parse_prediction("[1, 2]") is None
