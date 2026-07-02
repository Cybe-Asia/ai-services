"""Generate synthetic Indonesian transfer receipts for harness smoke tests.

These are fake receipts (fake names, banks' visual style only approximated)
so the eval pipeline can be validated end-to-end without touching real
parent data. Real accuracy numbers must come from staging receipts; this
set only proves the harness works and gives a rough upper bound on easy
cases plus a few deliberately degraded ones.

Usage:
    .venv/bin/python -m evals.receipt_extraction.synth \
        --out evals/receipt_extraction/data/synthetic
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size=size, index=1 if bold else 0)
    except OSError:
        return ImageFont.load_default()


THEMES = {
    "bca": {
        "header": (0, 60, 140),
        "header_text": "m-BCA",
        "title": "m-Transfer BERHASIL",
        "labels": {
            "date": "Tanggal",
            "sender": "Dari",
            "recipient": "Ke Rekening Tujuan",
            "amount": "Jumlah",
            "reference": "No. Referensi",
        },
    },
    "mandiri": {
        "header": (16, 33, 68),
        "header_text": "Livin' by Mandiri",
        "title": "Transaksi Berhasil",
        "labels": {
            "date": "Waktu Transaksi",
            "sender": "Sumber Dana",
            "recipient": "Penerima",
            "amount": "Nominal Transfer",
            "reference": "No. Transaksi",
        },
    },
    "seabank": {
        "header": (238, 77, 45),
        "header_text": "SeaBank",
        "title": "Transfer Berhasil",
        "labels": {
            "date": "Waktu",
            "sender": "Pengirim",
            "recipient": "Penerima",
            "amount": "Jumlah Transfer",
            "reference": "No. Referensi",
        },
    },
    "qris": {
        "header": (17, 143, 220),
        "header_text": "DANA",
        "title": "Pembayaran QRIS Berhasil",
        "labels": {
            "date": "Tanggal",
            "sender": "Nama Pengirim",
            "recipient": "Merchant",
            "amount": "Total Bayar",
            "reference": "ID Transaksi",
        },
    },
    "bni": {
        "header": (245, 130, 32),
        "header_text": "BNI Mobile Banking",
        "title": "Transfer Sukses",
        "labels": {
            "date": "Tanggal Transaksi",
            "sender": "Rekening Sumber",
            "recipient": "Rekening Tujuan",
            "amount": "Nominal",
            "reference": "Nomor Referensi",
        },
    },
    "bri": {
        "header": (0, 80, 157),
        "header_text": "BRImo",
        "title": "Transaksi Berhasil",
        "labels": {
            "date": "Tanggal",
            "sender": "Sumber Dana",
            "recipient": "Tujuan",
            "amount": "Nominal",
            "reference": "No. Ref",
        },
    },
}

# Each case: theme, displayed strings (varied formats on purpose), canonical truth,
# optional degradation, optional extra lines (e.g. the admin-fee trap).
CASES = [
    {
        "name": "bca_clean",
        "theme": "bca",
        "display": {
            "date": "28 Juni 2026 09:41:22",
            "sender": "BUDI SANTOSO",
            "recipient": "YAYASAN DIGITAL SCHOOLS",
            "amount": "Rp1.500.000,00",
            "reference": "95031428796541",
        },
        "truth": {
            "amount": 1500000,
            "currency": "IDR",
            "transfer_date": "2026-06-28",
            "sender_name": "Budi Santoso",
            "sender_bank": "BCA",
            "reference": "95031428796541",
        },
    },
    {
        "name": "mandiri_clean",
        "theme": "mandiri",
        "display": {
            "date": "25/06/2026 14:07:55",
            "sender": "SITI RAHAYU - Mandiri Tabungan",
            "recipient": "YAYASAN DIGITAL SCHOOLS",
            "amount": "Rp 2.750.000",
            "reference": "2506251407TRF88123",
        },
        "truth": {
            "amount": 2750000,
            "currency": "IDR",
            "transfer_date": "2026-06-25",
            "sender_name": "Siti Rahayu",
            "sender_bank": "Mandiri",
            "reference": "2506251407TRF88123",
        },
    },
    {
        "name": "seabank_clean",
        "theme": "seabank",
        "display": {
            "date": "30 Jun 2026, 19:23",
            "sender": "Agus Wijaya",
            "recipient": "Yayasan Digital Schools",
            "amount": "Rp500.000",
            "reference": "SB20260630112233",
        },
        "truth": {
            "amount": 500000,
            "currency": "IDR",
            "transfer_date": "2026-06-30",
            "sender_name": "Agus Wijaya",
            "sender_bank": "SeaBank",
            "reference": "SB20260630112233",
        },
    },
    {
        "name": "qris_dana",
        "theme": "qris",
        "display": {
            "date": "01 Juli 2026 15:30",
            "sender": "Dewi Lestari",
            "recipient": "DIGITAL SCHOOLS PPDB",
            "amount": "Rp250.000",
            "reference": "QR0107261530448899",
        },
        "truth": {
            "amount": 250000,
            "currency": "IDR",
            "transfer_date": "2026-07-01",
            "sender_name": "Dewi Lestari",
            "sender_bank": "DANA",
            "reference": "QR0107261530448899",
        },
    },
    {
        "name": "bni_clean",
        "theme": "bni",
        "display": {
            "date": "2026-06-27 08:14:35",
            "sender": "RINA KUSUMA",
            "recipient": "YAYASAN DIGITAL SCHOOLS",
            "amount": "IDR 1,750,000.00",
            "reference": "IB20260627556677",
        },
        "truth": {
            "amount": 1750000,
            "currency": "IDR",
            "transfer_date": "2026-06-27",
            "sender_name": "Rina Kusuma",
            "sender_bank": "BNI",
            "reference": "IB20260627556677",
        },
    },
    {
        "name": "brimo_clean",
        "theme": "bri",
        "display": {
            "date": "29 Juni 2026 08:14",
            "sender": "JOKO PRASETYO",
            "recipient": "YAYASAN DIGITAL SCHOOLS",
            "amount": "Rp3.200.000,00",
            "reference": "BRI2906260814352211",
        },
        "truth": {
            "amount": 3200000,
            "currency": "IDR",
            "transfer_date": "2026-06-29",
            "sender_name": "Joko Prasetyo",
            "sender_bank": "BRI",
            "reference": "BRI2906260814352211",
        },
    },
    {
        "name": "bca_admin_fee_trap",
        "theme": "bca",
        "display": {
            "date": "26 Juni 2026 11:05:10",
            "sender": "MAYA SARI",
            "recipient": "YAYASAN DIGITAL SCHOOLS",
            "amount": "Rp1.000.000,00",
            "reference": "95031429911223",
        },
        "extra_lines": [("Biaya Admin", "Rp2.500,00"), ("Total Debet", "Rp1.002.500,00")],
        "truth": {
            "amount": 1000000,
            "currency": "IDR",
            "transfer_date": "2026-06-26",
            "sender_name": "Maya Sari",
            "sender_bank": "BCA",
            "reference": "95031429911223",
        },
    },
    {
        "name": "bca_blur",
        "theme": "bca",
        "degrade": "blur",
        "display": {
            "date": "24 Juni 2026 16:48:03",
            "sender": "ANDI SAPUTRA",
            "recipient": "YAYASAN DIGITAL SCHOOLS",
            "amount": "Rp1.250.000,00",
            "reference": "95031427733489",
        },
        "truth": {
            "amount": 1250000,
            "currency": "IDR",
            "transfer_date": "2026-06-24",
            "sender_name": "Andi Saputra",
            "sender_bank": "BCA",
            "reference": "95031427733489",
        },
    },
    {
        "name": "mandiri_lowres",
        "theme": "mandiri",
        "degrade": "lowres",
        "display": {
            "date": "22/06/2026 10:31:44",
            "sender": "FITRI HANDAYANI - Mandiri Tabungan",
            "recipient": "YAYASAN DIGITAL SCHOOLS",
            "amount": "Rp 950.000",
            "reference": "2206261031TRF55678",
        },
        "truth": {
            "amount": 950000,
            "currency": "IDR",
            "transfer_date": "2026-06-22",
            "sender_name": "Fitri Handayani",
            "sender_bank": "Mandiri",
            "reference": "2206261031TRF55678",
        },
    },
    {
        "name": "seabank_photo",
        "theme": "seabank",
        "degrade": "photo",
        "display": {
            "date": "21 Jun 2026, 13:59",
            "sender": "Hendra Gunawan",
            "recipient": "Yayasan Digital Schools",
            "amount": "Rp4.500.000",
            "reference": "SB20260621998811",
        },
        "truth": {
            "amount": 4500000,
            "currency": "IDR",
            "transfer_date": "2026-06-21",
            "sender_name": "Hendra Gunawan",
            "sender_bank": "SeaBank",
            "reference": "SB20260621998811",
        },
    },
]

WIDTH, HEIGHT = 620, 900


def render_receipt(case: dict) -> Image.Image:
    theme = THEMES[case["theme"]]
    labels = theme["labels"]
    display = case["display"]

    image = Image.new("RGB", (WIDTH, HEIGHT), (245, 246, 248))
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, WIDTH, 110], fill=theme["header"])
    draw.text((30, 38), theme["header_text"], font=_font(34, bold=True), fill=(255, 255, 255))

    draw.rounded_rectangle([24, 140, WIDTH - 24, HEIGHT - 60], radius=16, fill=(255, 255, 255))
    draw.ellipse([WIDTH // 2 - 34, 170, WIDTH // 2 + 34, 238], fill=(46, 174, 96))
    draw.line([WIDTH // 2 - 14, 204, WIDTH // 2 - 3, 218], fill=(255, 255, 255), width=6)
    draw.line([WIDTH // 2 - 3, 218, WIDTH // 2 + 17, 190], fill=(255, 255, 255), width=6)
    draw.text(
        (WIDTH // 2, 268), theme["title"], font=_font(26, bold=True), fill=(30, 30, 30), anchor="mm"
    )
    draw.text(
        (WIDTH // 2, 306), display["date"], font=_font(19), fill=(110, 110, 110), anchor="mm"
    )

    rows = [
        (labels["amount"], display["amount"], True),
        (labels["sender"], display["sender"], False),
        (labels["recipient"], display["recipient"], False),
        (labels["reference"], display["reference"], False),
    ]
    for extra_label, extra_value in case.get("extra_lines", []):
        rows.append((extra_label, extra_value, False))

    y = 370
    for label, value, emphasize in rows:
        draw.text((60, y), label, font=_font(18), fill=(130, 130, 130))
        draw.text(
            (60, y + 28),
            value,
            font=_font(26 if emphasize else 21, bold=emphasize),
            fill=(20, 20, 20),
        )
        y += 92
        draw.line([60, y - 18, WIDTH - 60, y - 18], fill=(235, 235, 235), width=1)

    return image


def degrade(image: Image.Image, mode: str) -> Image.Image:
    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.6))
    if mode == "lowres":
        small = image.resize((WIDTH // 3, HEIGHT // 3), Image.BILINEAR)
        buffer = io.BytesIO()
        small.resize((WIDTH, HEIGHT), Image.BILINEAR).save(buffer, "JPEG", quality=30)
        return Image.open(buffer).convert("RGB")
    if mode == "photo":
        rotated = image.rotate(3.5, expand=True, fillcolor=(90, 88, 86))
        background = Image.new("RGB", rotated.size, (90, 88, 86))
        background.paste(rotated, (0, 0))
        buffer = io.BytesIO()
        background.save(buffer, "JPEG", quality=62)
        return Image.open(buffer).convert("RGB")
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic receipt eval set")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []
    for case in CASES:
        image = render_receipt(case)
        image = degrade(image, case.get("degrade", "none"))
        filename = f"{case['name']}.png"
        image.save(images_dir / filename)
        manifest_lines.append(
            json.dumps({"image": f"images/{filename}", "truth": case["truth"]}, ensure_ascii=False)
        )

    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(manifest_lines) + "\n")
    print(f"wrote {len(CASES)} receipts to {images_dir} and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
