"""Generate synthetic Indonesian transfer receipts for harness smoke tests.

These are fake receipts (fake names, banks' visual style only approximated)
so the eval pipeline can be validated end-to-end without touching real
parent data. This set is the interim decision gate while real receipts are
unavailable: it deliberately includes hard cases (WhatsApp recompression,
low resolution, skewed photos, dark mode, admin-fee traps) and non-receipt
distractors (including a chat screenshot that mentions an amount). Real
accuracy still needs real receipts — post-launch, finance corrections
accumulate exactly that labeled data.

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
WIDTH, HEIGHT = 620, 900


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
    "dana": {
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
    "gopay": {
        "header": (0, 170, 19),
        "header_text": "GoPay",
        "title": "Pembayaran Berhasil",
        "labels": {
            "date": "Tanggal",
            "sender": "Dari",
            "recipient": "Kepada",
            "amount": "Jumlah",
            "reference": "ID Transaksi",
        },
    },
    "ovo": {
        "header": (77, 26, 127),
        "header_text": "OVO",
        "title": "Transfer Berhasil",
        "labels": {
            "date": "Waktu",
            "sender": "Pengirim",
            "recipient": "Penerima",
            "amount": "Nominal",
            "reference": "No. Referensi",
        },
    },
    "shopeepay": {
        "header": (238, 77, 45),
        "header_text": "ShopeePay",
        "title": "Pembayaran Berhasil",
        "labels": {
            "date": "Waktu",
            "sender": "Dari",
            "recipient": "Merchant",
            "amount": "Total",
            "reference": "No. Transaksi",
        },
    },
    "bsi": {
        "header": (0, 130, 120),
        "header_text": "BSI Mobile",
        "title": "Transfer Berhasil",
        "labels": {
            "date": "Tanggal",
            "sender": "Rekening Asal",
            "recipient": "Rekening Tujuan",
            "amount": "Nominal",
            "reference": "No. Referensi",
        },
    },
    "permata": {
        "header": (0, 90, 60),
        "header_text": "PermataMobile X",
        "title": "Transaksi Berhasil",
        "labels": {
            "date": "Tanggal",
            "sender": "Dari Rekening",
            "recipient": "Ke Rekening",
            "amount": "Jumlah",
            "reference": "No. Referensi",
        },
    },
    "cimb": {
        "header": (120, 0, 30),
        "header_text": "OCTO Mobile",
        "title": "Transfer Berhasil",
        "labels": {
            "date": "Waktu",
            "sender": "Sumber Dana",
            "recipient": "Penerima",
            "amount": "Jumlah",
            "reference": "No. Referensi",
        },
    },
    "jago": {
        "header": (255, 122, 0),
        "header_text": "Jago",
        "title": "Transfer berhasil",
        "labels": {
            "date": "Tanggal",
            "sender": "Dari Kantong",
            "recipient": "Penerima",
            "amount": "Jumlah",
            "reference": "No. Referensi",
        },
    },
}

RECIPIENT = "YAYASAN DIGITAL SCHOOLS"

# One record per fake payer. `display` strings vary format on purpose;
# `truth` holds the canonical values the model must recover.
RECORDS = {
    "bca_budi": {
        "theme": "bca",
        "display": {
            "date": "28 Juni 2026 09:41:22",
            "sender": "BUDI SANTOSO",
            "amount": "Rp1.500.000,00",
            "reference": "95031428796541",
        },
        "truth": {
            "amount": 1500000,
            "transfer_date": "2026-06-28",
            "sender_name": "Budi Santoso",
            "sender_bank": "BCA",
            "reference": "95031428796541",
        },
    },
    "mandiri_siti": {
        "theme": "mandiri",
        "display": {
            "date": "25/06/2026 14:07:55",
            "sender": "SITI RAHAYU - Mandiri Tabungan",
            "amount": "Rp 2.750.000",
            "reference": "2506251407TRF88123",
        },
        "truth": {
            "amount": 2750000,
            "transfer_date": "2026-06-25",
            "sender_name": "Siti Rahayu",
            "sender_bank": "Mandiri",
            "reference": "2506251407TRF88123",
        },
    },
    "seabank_agus": {
        "theme": "seabank",
        "display": {
            "date": "30 Jun 2026, 19:23",
            "sender": "Agus Wijaya",
            "amount": "Rp500.000",
            "reference": "SB20260630112233",
        },
        "truth": {
            "amount": 500000,
            "transfer_date": "2026-06-30",
            "sender_name": "Agus Wijaya",
            "sender_bank": "SeaBank",
            "reference": "SB20260630112233",
        },
    },
    "dana_dewi": {
        "theme": "dana",
        "display": {
            "date": "01 Juli 2026 15:30",
            "sender": "Dewi Lestari",
            "amount": "Rp250.000",
            "reference": "QR0107261530448899",
        },
        "truth": {
            "amount": 250000,
            "transfer_date": "2026-07-01",
            "sender_name": "Dewi Lestari",
            "sender_bank": "DANA",
            "reference": "QR0107261530448899",
        },
    },
    "bni_rina": {
        "theme": "bni",
        "display": {
            "date": "2026-06-27 08:14:35",
            "sender": "RINA KUSUMA",
            "amount": "IDR 1,750,000.00",
            "reference": "IB20260627556677",
        },
        "truth": {
            "amount": 1750000,
            "transfer_date": "2026-06-27",
            "sender_name": "Rina Kusuma",
            "sender_bank": "BNI",
            "reference": "IB20260627556677",
        },
    },
    "bri_joko": {
        "theme": "bri",
        "display": {
            "date": "29 Juni 2026 08:14",
            "sender": "JOKO PRASETYO",
            "amount": "Rp3.200.000,00",
            "reference": "BRI2906260814352211",
        },
        "truth": {
            "amount": 3200000,
            "transfer_date": "2026-06-29",
            "sender_name": "Joko Prasetyo",
            "sender_bank": "BRI",
            "reference": "BRI2906260814352211",
        },
    },
    "bca_maya_fee": {
        "theme": "bca",
        "display": {
            "date": "26 Juni 2026 11:05:10",
            "sender": "MAYA SARI",
            "amount": "Rp1.000.000,00",
            "reference": "95031429911223",
        },
        "extra_lines": [("Biaya Admin", "Rp2.500,00"), ("Total Debet", "Rp1.002.500,00")],
        "truth": {
            "amount": 1000000,
            "transfer_date": "2026-06-26",
            "sender_name": "Maya Sari",
            "sender_bank": "BCA",
            "reference": "95031429911223",
        },
    },
    "gopay_andi": {
        "theme": "gopay",
        "display": {
            "date": "24 Jun 2026 16:48",
            "sender": "Andi Saputra",
            "amount": "Rp1.250.000",
            "reference": "GP-24062026-773348",
        },
        "truth": {
            "amount": 1250000,
            "transfer_date": "2026-06-24",
            "sender_name": "Andi Saputra",
            "sender_bank": "GoPay",
            "reference": "GP-24062026-773348",
        },
    },
    "ovo_fitri": {
        "theme": "ovo",
        "display": {
            "date": "22 Juni 2026 10:31",
            "sender": "Fitri Handayani",
            "amount": "Rp950.000",
            "reference": "OVO2206261031556",
        },
        "truth": {
            "amount": 950000,
            "transfer_date": "2026-06-22",
            "sender_name": "Fitri Handayani",
            "sender_bank": "OVO",
            "reference": "OVO2206261031556",
        },
    },
    "shopeepay_hendra": {
        "theme": "shopeepay",
        "display": {
            "date": "21 Jun 2026 13:59",
            "sender": "Hendra Gunawan",
            "amount": "Rp4.500.000",
            "reference": "SP20260621998811",
        },
        "truth": {
            "amount": 4500000,
            "transfer_date": "2026-06-21",
            "sender_name": "Hendra Gunawan",
            "sender_bank": "ShopeePay",
            "reference": "SP20260621998811",
        },
    },
    "bsi_nurul": {
        "theme": "bsi",
        "dark": True,
        "display": {
            "date": "23 Juni 2026 09:12",
            "sender": "NURUL AINI",
            "amount": "Rp2.000.000,00",
            "reference": "BSI2306260912344",
        },
        "truth": {
            "amount": 2000000,
            "transfer_date": "2026-06-23",
            "sender_name": "Nurul Aini",
            "sender_bank": "BSI",
            "reference": "BSI2306260912344",
        },
    },
    "permata_rizky": {
        "theme": "permata",
        "display": {
            "date": "26/06/2026 17:45",
            "sender": "RIZKY RAMADHAN",
            "amount": "Rp 1.500.000,-",
            "reference": "PMT2606261745882",
        },
        "truth": {
            "amount": 1500000,
            "transfer_date": "2026-06-26",
            "sender_name": "Rizky Ramadhan",
            "sender_bank": "Permata",
            "reference": "PMT2606261745882",
        },
    },
    "cimb_putri": {
        "theme": "cimb",
        "display": {
            "date": "27 Jun 2026 11:20",
            "sender": "PUTRI MAHARANI",
            "amount": "Rp1.800.000,00",
            "reference": "CIMB2706261120774",
        },
        "truth": {
            "amount": 1800000,
            "transfer_date": "2026-06-27",
            "sender_name": "Putri Maharani",
            "sender_bank": "CIMB",
            "reference": "CIMB2706261120774",
        },
    },
    "jago_bayu": {
        "theme": "jago",
        "dark": True,
        "display": {
            "date": "20 Juni 2026 08:05",
            "sender": "Bayu Nugroho",
            "amount": "Rp600.000",
            "reference": "JG20062026080544",
        },
        "truth": {
            "amount": 600000,
            "transfer_date": "2026-06-20",
            "sender_name": "Bayu Nugroho",
            "sender_bank": "Jago",
            "reference": "JG20062026080544",
        },
    },
    "bca_lina": {
        "theme": "bca",
        "dark": True,
        "display": {
            "date": "19 Juni 2026 21:33",
            "sender": "LINA MARLINA",
            "amount": "Rp2.250.000,00",
            "reference": "95031426612378",
        },
        "truth": {
            "amount": 2250000,
            "transfer_date": "2026-06-19",
            "sender_name": "Lina Marlina",
            "sender_bank": "BCA",
            "reference": "95031426612378",
        },
    },
}

# Every record ships a clean render; these additionally ship a degraded copy.
DEGRADED_VARIANTS = [
    ("bca_budi", "whatsapp"),
    ("mandiri_siti", "lowres"),
    ("seabank_agus", "whatsapp"),
    ("dana_dewi", "photo"),
    ("bni_rina", "lowres"),
    ("bri_joko", "photo"),
    ("bca_maya_fee", "whatsapp"),
    ("gopay_andi", "blur"),
    ("ovo_fitri", "blur"),
    ("shopeepay_hendra", "photo"),
    ("permata_rizky", "whatsapp"),
    ("cimb_putri", "lowres"),
]


def render_receipt(record: dict) -> Image.Image:
    theme = THEMES[record["theme"]]
    labels = theme["labels"]
    display = record["display"]
    dark = bool(record.get("dark"))

    page = (18, 18, 20) if dark else (245, 246, 248)
    card = (30, 30, 34) if dark else (255, 255, 255)
    text_primary = (235, 235, 235) if dark else (20, 20, 20)
    text_secondary = (150, 150, 155) if dark else (110, 110, 110)
    text_label = (130, 130, 135) if dark else (130, 130, 130)
    divider = (55, 55, 60) if dark else (235, 235, 235)

    image = Image.new("RGB", (WIDTH, HEIGHT), page)
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, WIDTH, 110], fill=theme["header"])
    draw.text((30, 38), theme["header_text"], font=_font(34, bold=True), fill=(255, 255, 255))

    draw.rounded_rectangle([24, 140, WIDTH - 24, HEIGHT - 60], radius=16, fill=card)
    draw.ellipse([WIDTH // 2 - 34, 170, WIDTH // 2 + 34, 238], fill=(46, 174, 96))
    draw.line([WIDTH // 2 - 14, 204, WIDTH // 2 - 3, 218], fill=(255, 255, 255), width=6)
    draw.line([WIDTH // 2 - 3, 218, WIDTH // 2 + 17, 190], fill=(255, 255, 255), width=6)
    draw.text(
        (WIDTH // 2, 268),
        theme["title"],
        font=_font(26, bold=True),
        fill=text_primary,
        anchor="mm",
    )
    draw.text((WIDTH // 2, 306), display["date"], font=_font(19), fill=text_secondary, anchor="mm")

    rows = [
        (labels["amount"], display["amount"], True),
        (labels["sender"], display["sender"], False),
        (labels["recipient"], RECIPIENT, False),
        (labels["reference"], display["reference"], False),
    ]
    for extra_label, extra_value in record.get("extra_lines", []):
        rows.append((extra_label, extra_value, False))

    y = 370
    for label, value, emphasize in rows:
        draw.text((60, y), label, font=_font(18), fill=text_label)
        draw.text(
            (60, y + 28),
            value,
            font=_font(26 if emphasize else 21, bold=emphasize),
            fill=text_primary,
        )
        y += 92
        draw.line([60, y - 18, WIDTH - 60, y - 18], fill=divider, width=1)

    return image


def degrade(image: Image.Image, mode: str) -> Image.Image:
    if mode == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=1.6))
    if mode == "lowres":
        small = image.resize((WIDTH // 3, HEIGHT // 3), Image.BILINEAR)
        buffer = io.BytesIO()
        small.resize((WIDTH, HEIGHT), Image.BILINEAR).save(buffer, "JPEG", quality=30)
        return Image.open(buffer).convert("RGB")
    if mode == "whatsapp":
        small = image.resize((int(WIDTH * 0.62), int(HEIGHT * 0.62)), Image.BILINEAR)
        buffer = io.BytesIO()
        small.save(buffer, "JPEG", quality=45)
        once = Image.open(buffer).convert("RGB")
        buffer = io.BytesIO()
        once.save(buffer, "JPEG", quality=35)
        return Image.open(buffer).convert("RGB")
    if mode == "photo":
        rotated = image.rotate(3.5, expand=True, fillcolor=(90, 88, 86))
        background = Image.new("RGB", rotated.size, (90, 88, 86))
        background.paste(rotated, (0, 0))
        buffer = io.BytesIO()
        background.save(buffer, "JPEG", quality=62)
        return Image.open(buffer).convert("RGB")
    return image


def render_chat_screenshot() -> Image.Image:
    """WhatsApp-style chat that MENTIONS a transfer amount but is not a receipt."""
    image = Image.new("RGB", (WIDTH, HEIGHT), (229, 221, 213))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 96], fill=(7, 94, 84))
    draw.text((28, 34), "Admin Digital Schools", font=_font(24, bold=True), fill=(255, 255, 255))

    bubbles = [
        ("in", "Selamat siang Bu, apakah pembayaran formulir sudah dilakukan?"),
        ("out", "Sudah pak, saya transfer Rp1.500.000 tadi pagi ya"),
        ("out", "Nanti bukti transfernya saya kirim menyusul"),
        ("in", "Baik Bu, kami tunggu bukti transfernya"),
    ]
    y = 140
    for direction, message in bubbles:
        lines: list[str] = []
        line = ""
        for word in message.split(" "):
            candidate = f"{line} {word}".strip()
            if len(candidate) > 34:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
        height = 26 * len(lines) + 24
        width = min(430, max(len(text) for text in lines) * 11 + 40)
        x0 = WIDTH - width - 28 if direction == "out" else 28
        fill = (220, 248, 198) if direction == "out" else (255, 255, 255)
        draw.rounded_rectangle([x0, y, x0 + width, y + height], radius=12, fill=fill)
        for index, text in enumerate(lines):
            draw.text((x0 + 18, y + 12 + index * 26), text, font=_font(18), fill=(30, 30, 30))
        y += height + 18
    return image


def render_settings_screenshot() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (28, 28, 30))
    draw = ImageDraw.Draw(image)
    rows = [
        "Akun saya",
        "Notifikasi",
        "Privasi",
        "Bahasa",
        "Tema gelap",
        "Bantuan",
        "Tentang aplikasi",
        "Keluar",
    ]
    y = 60
    for label in rows:
        draw.text((40, y), label, font=_font(24), fill=(235, 235, 235))
        draw.line([40, y + 52, WIDTH - 40, y + 52], fill=(55, 55, 58), width=1)
        y += 92
    return image


def render_document_photo() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (90, 88, 86))
    draw = ImageDraw.Draw(image)
    draw.rectangle([70, 90, WIDTH - 70, HEIGHT - 90], fill=(250, 249, 245))
    draw.text((100, 130), "SURAT PERNYATAAN", font=_font(24, bold=True), fill=(40, 40, 40))
    y = 190
    widths = [420, 400, 430, 380, 410, 300, 420, 390, 260, 415, 405, 340]
    for line_width in widths:
        draw.rectangle([100, y, 100 + line_width, y + 10], fill=(120, 118, 115))
        y += 34
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=60)
    return Image.open(buffer).convert("RGB")


NON_RECEIPTS = {
    "nonreceipt_chat_mentions_amount": render_chat_screenshot,
    "nonreceipt_settings_screen": render_settings_screenshot,
    "nonreceipt_document_photo": render_document_photo,
}

NULL_TRUTH = {
    "is_receipt": False,
    "amount": None,
    "currency": None,
    "transfer_date": None,
    "sender_name": None,
    "sender_bank": None,
    "reference": None,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic receipt eval set")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = []

    def emit(name: str, image: Image.Image, truth: dict) -> None:
        filename = f"{name}.png"
        image.save(images_dir / filename)
        manifest_lines.append(
            json.dumps({"image": f"images/{filename}", "truth": truth}, ensure_ascii=False)
        )

    for key, record in RECORDS.items():
        truth = {"is_receipt": True, "currency": "IDR", **record["truth"]}
        emit(f"{key}_clean", render_receipt(record), truth)

    for key, mode in DEGRADED_VARIANTS:
        record = RECORDS[key]
        truth = {"is_receipt": True, "currency": "IDR", **record["truth"]}
        emit(f"{key}_{mode}", degrade(render_receipt(record), mode), truth)

    for name, renderer in NON_RECEIPTS.items():
        emit(name, renderer(), dict(NULL_TRUTH))

    manifest = out_dir / "manifest.jsonl"
    manifest.write_text("\n".join(manifest_lines) + "\n")
    total = len(RECORDS) + len(DEGRADED_VARIANTS) + len(NON_RECEIPTS)
    print(f"wrote {total} cases to {images_dir} and {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
