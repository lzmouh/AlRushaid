"""
Invoice OCR database for photographed restaurant invoices.

Designed around receipts like the supplied sample:
- English text is extracted; Arabic text is ignored.
- Multiple photos are loaded in one upload action.
- Images are processed ONE AT A TIME.
- After each image is processed, the SQLite database is updated and the
  invoice date / total summary table is refreshed immediately.
- No Tesseract is required.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rapidocr_onnxruntime import RapidOCR


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
IMAGE_DIR = DATA_DIR / "invoice_images"
DB_PATH = DATA_DIR / "invoices.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# OCR engine
# ---------------------------------------------------------------------

@st.cache_resource
def get_ocr_engine():
    """Load OCR models once per Streamlit process."""
    return RapidOCR()


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS invoices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                file_name TEXT NOT NULL,
                processed_at TEXT NOT NULL,

                invoice_date TEXT,
                printed_at TEXT,
                merchant TEXT,
                vat_number TEXT,
                order_number TEXT,
                check_number TEXT,
                customer TEXT,
                creator TEXT,
                closer TEXT,

                subtotal REAL,
                vat_rate REAL,
                vat_amount REAL,
                total_amount REAL,
                payment_method TEXT,
                products_count INTEGER,

                items_json TEXT,
                raw_ocr_text TEXT,
                ocr_confidence REAL,

                status TEXT NOT NULL,
                error_message TEXT
            )
            """
        )


def save_invoice(record: dict[str, Any]) -> bool:
    """Insert one invoice. Returns False if the same file hash already exists."""
    columns = [
        "file_hash", "file_name", "processed_at",
        "invoice_date", "printed_at", "merchant", "vat_number",
        "order_number", "check_number", "customer", "creator", "closer",
        "subtotal", "vat_rate", "vat_amount", "total_amount",
        "payment_method", "products_count", "items_json", "raw_ocr_text",
        "ocr_confidence", "status", "error_message",
    ]

    placeholders = ",".join(["?"] * len(columns))
    values = [record.get(c) for c in columns]

    with sqlite3.connect(DB_PATH) as con:
        cur = con.execute(
            f"""
            INSERT OR IGNORE INTO invoices ({",".join(columns)})
            VALUES ({placeholders})
            """,
            values,
        )
        return cur.rowcount == 1


def load_summary() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            """
            SELECT
                id AS "#",
                file_name AS "Photo",
                invoice_date AS "Invoice Date",
                total_amount AS "Total Amount",
                vat_amount AS "VAT",
                status AS "Status"
            FROM invoices
            ORDER BY id DESC
            """,
            con,
        )


# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()

    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    return rect


def four_point_warp(image: np.ndarray) -> np.ndarray:
    """
    Try to detect the receipt as the largest quadrilateral.
    If detection is unreliable, return the original image.
    """
    h, w = image.shape[:2]
    scale = 1200.0 / max(h, w)
    small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = small.shape[0] * small.shape[1]
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < image_area * 0.10:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            candidates.append((area, approx.reshape(4, 2)))

    if not candidates:
        return image

    _, pts = max(candidates, key=lambda x: x[0])

    # Convert coordinates back to original image.
    if scale < 1:
        pts = pts / scale

    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    if max_width < 300 or max_height < 300:
        return image

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped


def make_ocr_variants(image: np.ndarray) -> list[np.ndarray]:
    """
    Produce several OCR-friendly versions.
    The OCR engine will be run sequentially on these variants.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast.
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Upscaling is important for small receipt characters.
    scale = 2.5
    up_gray = cv2.resize(
        gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
    up_contrast = cv2.resize(
        contrast, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )

    # Gentle sharpening.
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(up_contrast, -1, kernel)

    # Adaptive threshold helps thermal-paper receipts.
    adaptive = cv2.adaptiveThreshold(
        up_contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return [
        up_gray,
        up_contrast,
        sharp,
        adaptive,
    ]


# ---------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------

ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def clean_english(text: str) -> str:
    """
    Remove Arabic/non-ASCII OCR output while retaining English, numbers,
    punctuation and currency-independent receipt information.
    """
    text = ASCII_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def run_ocr(engine: RapidOCR, image: np.ndarray) -> tuple[str, float]:
    result, _ = engine(image)

    if not result:
        return "", 0.0

    pieces: list[str] = []
    scores: list[float] = []

    for row in result:
        # RapidOCR returns [box, text, confidence].
        if len(row) < 3:
            continue

        text = clean_english(str(row[1]))
        try:
            score = float(row[2])
        except Exception:
            score = 0.0

        if text:
            pieces.append(text)
            scores.append(score)

    return "\n".join(pieces), (float(np.mean(scores)) if scores else 0.0)


def score_ocr_text(text: str, confidence: float) -> float:
    """
    Prefer OCR variants that both have good confidence and contain
    invoice-specific anchor words.
    """
    low = text.lower()

    anchors = [
        "total", "subtotal", "vat", "printed", "order",
        "check", "customer", "price", "qty", "restaurant",
    ]
    anchor_hits = sum(1 for a in anchors if a in low)

    # Confidence dominates; anchors break ties.
    return confidence * 100.0 + anchor_hits * 3.0 + min(len(text), 3000) / 3000.0


def ocr_image(image_bytes: bytes) -> tuple[str, float, np.ndarray]:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode the image.")

    corrected = four_point_warp(image)
    variants = make_ocr_variants(corrected)

    engine = get_ocr_engine()

    best_text = ""
    best_conf = 0.0
    best_score = -1.0

    for variant in variants:
        text, conf = run_ocr(engine, variant)
        score = score_ocr_text(text, conf)

        if score > best_score:
            best_score = score
            best_text = text
            best_conf = conf

    return best_text, best_conf, corrected


# ---------------------------------------------------------------------
# Structured extraction
# ---------------------------------------------------------------------

DATE_RE = re.compile(r"\b(20\d{2})[/-](\d{1,2})[/-](\d{1,2})\b")
AMOUNT_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})(?!\d)"
)


def normalize_amount(value: str) -> float:
    return float(value.replace(",", ""))


def amounts_in(line: str) -> list[float]:
    return [normalize_amount(x) for x in AMOUNT_RE.findall(line)]


def first_date(text: str) -> str | None:
    m = DATE_RE.search(text)
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"


def extract_after_label(lines: list[str], labels: list[str]) -> str | None:
    for i, line in enumerate(lines):
        low = line.lower()
        for label in labels:
            if label in low:
                # Prefer content after the label on the same line.
                parts = re.split(re.escape(label), line, maxsplit=1, flags=re.I)
                if len(parts) == 2:
                    value = parts[1].strip(" :#-")
                    if value:
                        return value

                # Otherwise inspect the next line.
                if i + 1 < len(lines):
                    value = lines[i + 1].strip()
                    if value:
                        return value
    return None


def extract_labeled_amount(lines: list[str], label: str) -> float | None:
    for line in lines:
        if label.lower() in line.lower():
            vals = amounts_in(line)
            if vals:
                return vals[-1]
    return None


def extract_vat_rate(lines: list[str]) -> float | None:
    for line in lines:
        if "vat" in line.lower():
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if m:
                return float(m.group(1))
    return None


def extract_total(lines: list[str]) -> float | None:
    # "Total" must be checked before "Subtotal".
    for line in lines:
        low = line.lower()
        if "total" in low and "subtotal" not in low:
            vals = amounts_in(line)
            if vals:
                return vals[-1]
    return None


def extract_items(lines: list[str]) -> list[dict[str, Any]]:
    """
    Extract receipt lines shaped approximately like:
      2 CHICKEN & PINEAPPLE PIZZA LARGE 96.00
    """
    items = []

    for line in lines:
        cleaned = line.strip()
        low = cleaned.lower()

        if any(
            x in low
            for x in [
                "subtotal", "vat", "total", "payment", "products count",
                "printed", "creator", "closer", "customer", "order",
                "check", "invoice", "tax invoice", "thank you",
            ]
        ):
            continue

        # Qty + description + amount.
        m = re.match(
            r"^\s*(\d+)\s+(.+?)\s+"
            r"(\d{1,3}(?:,\d{3})*\.\d{2}|\d+\.\d{2})\s*$",
            cleaned,
            flags=re.I,
        )
        if not m:
            continue

        qty = int(m.group(1))
        description = clean_english(m.group(2)).strip()
        amount = normalize_amount(m.group(3))

        if description:
            items.append(
                {
                    "qty": qty,
                    "item": description,
                    "line_total": amount,
                }
            )

    return items


def parse_invoice(raw_text: str, confidence: float) -> dict[str, Any]:
    lines = [x.strip() for x in raw_text.splitlines() if x.strip()]

    # Normalize OCR spacing without changing useful punctuation.
    lines = [re.sub(r"\s+", " ", x) for x in lines]

    invoice_date = first_date(raw_text)

    printed_at = None
    for line in lines:
        if "printed at" in line.lower():
            m = re.search(
                r"(20\d{2}[/-]\d{1,2}[/-]\d{1,2})\s+"
                r"(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)?",
                line,
                flags=re.I,
            )
            if m:
                printed_at = " ".join(x for x in m.groups() if x)
                break

    merchant = None
    for line in lines:
        low = line.lower()
        if "real estate investment company" in low:
            merchant = line
            break

    vat_number = None
    for line in lines:
        if "vat:" in line.lower() or re.search(r"\bvat\b", line.lower()):
            m = re.search(r"\b(\d{10,20})\b", line.replace(" ", ""))
            if m:
                vat_number = m.group(1)
                break

    order_number = None
    for line in lines:
        m = re.search(r"order\s*#?\s*(\d+)", line, flags=re.I)
        if m:
            order_number = m.group(1)
            break

    check_number = None
    for line in lines:
        m = re.search(r"check\s*#?\s*(\d+)", line, flags=re.I)
        if m:
            check_number = m.group(1)
            break

    creator = extract_after_label(lines, ["creator"])
    closer = extract_after_label(lines, ["closer"])
    customer = extract_after_label(lines, ["customer"])

    payment_method = None
    for line in lines:
        if "payment" in line.lower():
            payment_method = re.sub(
                r"^\s*payment\s*[-:]*\s*",
                "",
                line,
                flags=re.I,
            ).strip()
            break

    subtotal = extract_labeled_amount(lines, "subtotal")
    vat_amount = extract_labeled_amount(lines, "vat")
    total_amount = extract_total(lines)
    vat_rate = extract_vat_rate(lines)

    products_count = None
    for line in lines:
        m = re.search(r"products?\s*count\s*(\d+)", line, flags=re.I)
        if m:
            products_count = int(m.group(1))
            break

    items = extract_items(lines)

    # If product count wasn't recognized, use the sum of quantities.
    if products_count is None and items:
        products_count = sum(x["qty"] for x in items)

    # Basic consistency recovery:
    # if total was missed but subtotal + VAT exists, calculate it.
    if total_amount is None and subtotal is not None and vat_amount is not None:
        total_amount = round(subtotal + vat_amount, 2)

    return {
        "invoice_date": invoice_date,
        "printed_at": printed_at,
        "merchant": merchant,
        "vat_number": vat_number,
        "order_number": order_number,
        "check_number": check_number,
        "customer": customer,
        "creator": creator,
        "closer": closer,
        "subtotal": subtotal,
        "vat_rate": vat_rate,
        "vat_amount": vat_amount,
        "total_amount": total_amount,
        "payment_method": payment_method,
        "products_count": products_count,
        "items_json": json.dumps(items, ensure_ascii=False),
        "raw_ocr_text": raw_text,
        "ocr_confidence": round(confidence, 4),
        "status": "OK" if total_amount is not None else "REVIEW",
        "error_message": None,
    }


# ---------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_original_image(file_hash: str, file_name: str, data: bytes) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", file_name)
    path = IMAGE_DIR / f"{file_hash[:12]}_{safe_name}"
    if not path.exists():
        path.write_bytes(data)
    return path


def process_one_file(uploaded_file) -> dict[str, Any]:
    data = uploaded_file.getvalue()
    file_hash = sha256_bytes(data)

    record = {
        "file_hash": file_hash,
        "file_name": uploaded_file.name,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }

    save_original_image(file_hash, uploaded_file.name, data)

    raw_text, confidence, _ = ocr_image(data)
    record.update(parse_invoice(raw_text, confidence))

    return record


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="Invoice OCR Database",
    page_icon="🧾",
    layout="wide",
)

init_db()

st.title("🧾 Invoice OCR Database")
st.caption(
    "Upload all invoice photos at once. They are processed sequentially, "
    "and the database/summary is updated after every photo."
)

uploaded_files = st.file_uploader(
    "Select invoice photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    help="All selected photos are loaded together, but OCR runs one image at a time.",
)

col1, col2 = st.columns([1, 3])
with col1:
    process_clicked = st.button(
        "▶ Process invoices",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )
with col2:
    if uploaded_files:
        st.info(f"{len(uploaded_files)} photo(s) loaded and ready.")

summary_placeholder = st.empty()
progress_placeholder = st.empty()
message_placeholder = st.empty()

# Always show the current database.
summary_placeholder.dataframe(
    load_summary(),
    use_container_width=True,
    hide_index=True,
)

if process_clicked and uploaded_files:
    progress = progress_placeholder.progress(0)
    results_box = st.container()

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        message_placeholder.info(
            f"Processing {index}/{len(uploaded_files)}: **{uploaded_file.name}**"
        )

        try:
            record = process_one_file(uploaded_file)

            inserted = save_invoice(record)

            # Immediately refresh the table after this ONE image.
            current = load_summary()
            summary_placeholder.dataframe(
                current,
                use_container_width=True,
                hide_index=True,
            )

            if inserted:
                status = record["status"]
                total = record.get("total_amount")
                date = record.get("invoice_date")

                with results_box:
                    if status == "OK":
                        st.success(
                            f"Processed **{uploaded_file.name}** — "
                            f"date: `{date or 'not found'}` — "
                            f"total: `{total if total is not None else 'not found'}`"
                        )
                    else:
                        st.warning(
                            f"Processed **{uploaded_file.name}**, but the total "
                            "could not be confidently extracted. Marked REVIEW."
                        )
            else:
                with results_box:
                    st.info(
                        f"Skipped duplicate: **{uploaded_file.name}** "
                        "(same file hash already exists in the database)."
                    )

        except Exception as exc:
            # Still record the failed image so it is visible in the database.
            failed = {
                **record,
                "invoice_date": None,
                "printed_at": None,
                "merchant": None,
                "vat_number": None,
                "order_number": None,
                "check_number": None,
                "customer": None,
                "creator": None,
                "closer": None,
                "subtotal": None,
                "vat_rate": None,
                "vat_amount": None,
                "total_amount": None,
                "payment_method": None,
                "products_count": None,
                "items_json": "[]",
                "raw_ocr_text": "",
                "ocr_confidence": 0.0,
                "status": "ERROR",
                "error_message": str(exc),
            }
            save_invoice(failed)

            summary_placeholder.dataframe(
                load_summary(),
                use_container_width=True,
                hide_index=True,
            )

            with results_box:
                st.error(f"Failed: {uploaded_file.name}: {exc}")

        progress.progress(index / len(uploaded_files))

    message_placeholder.success(
        f"Finished processing {len(uploaded_files)} photo(s)."
    )

# ---------------------------------------------------------------------
# Database export
# ---------------------------------------------------------------------

st.divider()
st.subheader("Database export")

summary = load_summary()

if not summary.empty:
    csv_data = summary.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download invoice summary CSV",
        data=csv_data,
        file_name="invoice_summary.csv",
        mime="text/csv",
    )

    st.caption(
        f"SQLite database: {DB_PATH}  |  "
        f"Stored invoice photos: {IMAGE_DIR}"
    )

st.divider()
st.caption(
    "OCR engine: RapidOCR/ONNX. Arabic OCR output is discarded; English, "
    "numbers and receipt punctuation are retained."
)
