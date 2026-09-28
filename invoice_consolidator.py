
import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import shutil
import os

# Optional PDF support
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


st.set_page_config(
    page_title="Restaurant Invoice Consolidator",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 Restaurant Invoice Consolidator")
st.caption(
    "Upload multiple restaurant invoices at once. The app OCRs each invoice, "
    "extracts the transaction fields and calculates year-to-date subtotals and totals."
)


# ---------------------------------------------------------------------
# OCR / image preparation
# ---------------------------------------------------------------------

def configure_tesseract():
    \"\"
    Streamlit Community Cloud installs system packages from packages.txt.
    This function also checks common Linux locations so the app gives a
    useful error if Tesseract was not installed.
    \"\"
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    return None


TESSERACT_PATH = configure_tesseract()

if not TESSERACT_PATH:
    st.error(
        "Tesseract OCR is not installed. "
        "For Streamlit Community Cloud, make sure the project contains "
        "a packages.txt file with 'tesseract-ocr' and 'tesseract-ocr-ara', "
        "then redeploy/reboot the app."
    )
    st.stop()


def preprocess_image(image: Image.Image) -> Image.Image:
    """Prepare a receipt image for OCR."""
    img = image.convert("RGB")

    # Upscale narrow thermal receipts.
    scale = max(1.0, 1800 / max(img.width, 1))
    if scale > 1:
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.Resampling.LANCZOS,
        )

    # Grayscale + contrast enhancement.
    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(2.0)
    gray = ImageEnhance.Sharpness(gray).enhance(1.5)

    # OpenCV adaptive thresholding works well with photographed receipts.
    arr = np.array(gray)
    arr = cv2.GaussianBlur(arr, (3, 3), 0)
    threshold = cv2.adaptiveThreshold(
        arr,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    return Image.fromarray(threshold)


def ocr_image(image: Image.Image) -> str:
    """Run OCR. English is always attempted; Arabic is used if installed."""
    processed = preprocess_image(image)

    # Use English + Arabic when Arabic Tesseract data is available.
    try:
        langs = pytesseract.get_languages(config="")
    except Exception:
        langs = ["eng"]

    lang = "eng+ara" if "ara" in langs else "eng"

    configs = [
        "--psm 6",
        "--psm 4",
        "--psm 11",
    ]

    results = []
    for config in configs:
        try:
            results.append(pytesseract.image_to_string(processed, lang=lang, config=config))
        except Exception:
            pass

    # Longest result is usually the most complete for receipts.
    return max(results, key=len, default="")


def pdf_to_images(data: bytes):
    """Render every page of a PDF to PIL images."""
    if not PDF_AVAILABLE:
        raise RuntimeError("PDF support requires PyMuPDF (pip install pymupdf).")

    document = fitz.open(stream=data, filetype="pdf")
    images = []

    for page in document:
        pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    return images


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

MONEY_RE = r"(?:[$€£¥₩﷼]|SAR|SR|ر\.?س\.?|﷼)?\s*([0-9]+(?:[.,][0-9]{1,2})?)"


def money_to_float(value):
    if value is None:
        return None

    s = str(value).strip()
    s = s.replace(",", "")

    # Keep only the numeric portion.
    match = re.search(r"-?\d+(?:\.\d+)?", s)
    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def normalize_text(text):
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def first_match(patterns, text, flags=re.I | re.M):
    for pattern in patterns:
        m = re.search(pattern, text, flags)
        if m:
            return m.group(1).strip()
    return None


def parse_date(text):
    patterns = [
        r"(\d{4}/\d{2}/\d{2})\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?",
        r"(\d{4}-\d{2}-\d{2})",
        r"(\d{4}/\d{2}/\d{2})",
    ]

    value = first_match(patterns, text)
    if not value:
        return None

    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


def parse_invoice(text, filename):
    text = normalize_text(text)

    invoice = {
        "filename": filename,
        "invoice_date": parse_date(text),
        "order_number": None,
        "check_number": None,
        "vat_number": None,
        "customer": None,
        "creator": None,
        "closer": None,
        "service_type": None,
        "subtotal": None,
        "vat": None,
        "total": None,
        "payment": None,
        "products_count": None,
        "eat_number": None,
        "currency": None,
        "raw_text": text,
    }

    invoice["order_number"] = first_match(
        [
            r"Order\s*#\s*([A-Za-z0-9-]+)",
            r"Order\s*No\.?\s*[:#]?\s*([A-Za-z0-9-]+)",
        ],
        text,
    )

    invoice["check_number"] = first_match(
        [
            r"Check\s*#\s*([A-Za-z0-9-]+)",
            r"Check\s*No\.?\s*[:#]?\s*([A-Za-z0-9-]+)",
        ],
        text,
    )

    invoice["vat_number"] = first_match(
        [
            r"VAT\s*[:#]?\s*([0-9]{10,20})",
            r"VAT\s*No\.?\s*[:#]?\s*([0-9]{10,20})",
        ],
        text,
    )

    invoice["customer"] = first_match(
        [
            r"Customer\s*:\s*(.+)",
            r"ustomer\s*:\s*(.+)",
        ],
        text,
    )

    invoice["creator"] = first_match([r"Creator\s*:\s*(.+)"], text)
    invoice["closer"] = first_match([r"Closer\s*:\s*(.+)"], text)

    invoice["service_type"] = first_match(
        [
            r"\b(Drive\s+Thru)\b",
            r"\b(Dine\s*In)\b",
            r"\b(Take\s*Away)\b",
            r"\b(Delivery)\b",
        ],
        text,
    )

    # The sample invoice uses:
    # Subtotal   120.87
    # VAT(15.0%) 18.13
    # Total      139.00
    invoice["subtotal"] = money_to_float(
        first_match(
            [
                r"Subtotal\s*[:\-]?\s*" + MONEY_RE,
            ],
            text,
        )
    )

    invoice["vat"] = money_to_float(
        first_match(
            [
                r"VAT\s*\(?\s*\d+(?:\.\d+)?%\s*\)?\s*[:\-]?\s*" + MONEY_RE,
                r"VAT\s*[:\-]?\s*" + MONEY_RE,
            ],
            text,
        )
    )

    invoice["total"] = money_to_float(
        first_match(
            [
                r"\bTotal\s*[:\-]?\s*" + MONEY_RE,
            ],
            text,
        )
    )

    invoice["payment"] = first_match(
        [
            r"Payment\s*-\s*(.+)",
            r"Payment\s*:\s*(.+)",
        ],
        text,
    )

    product_count = first_match(
        [
            r"Products\s+Count\s+(\d+)",
            r"Product\s+Count\s+(\d+)",
        ],
        text,
    )
    if product_count:
        invoice["products_count"] = int(product_count)

    invoice["eat_number"] = first_match(
        [
            r"Eat\s*No\.?\s*[:#]?\s*([A-Za-z0-9-]+)",
            r"Ext\s*No\.?\s*[:#]?\s*([A-Za-z0-9-]+)",
        ],
        text,
    )

    # Infer currency symbol from the OCR text where possible.
    for symbol, code in [
        ("₩", "KRW"),
        ("SAR", "SAR"),
        ("﷼", "SAR"),
        ("SR", "SAR"),
        ("$", "USD"),
        ("€", "EUR"),
        ("£", "GBP"),
    ]:
        if symbol in text:
            invoice["currency"] = code
            break

    return invoice


# ---------------------------------------------------------------------
# Item extraction
# ---------------------------------------------------------------------

def extract_items(text):
    """
    Attempts to identify receipt item lines.

    Typical sample:
        2  Tandoori flavor Chicken Pizza   90.00
        1  GRILLED CHICKEN BREAST          49.00

    The parser deliberately stops at subtotal/VAT/total lines.
    """
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    items = []

    stop_words = re.compile(
        r"^(Subtotal|VAT|Total|Payment|Official|Products?\s+Count|Thank|Eat\s+No|Ext\s+No)",
        re.I,
    )

    # quantity + description + price
    pattern = re.compile(
        r"^(\d+)\s+(.+?)\s+(?:[$€£¥₩﷼]|SAR|SR|ر\.?س\.?|﷼)?\s*"
        r"(\d+(?:[.,]\d{1,2})?)\s*$",
        re.I,
    )

    in_items = False

    for line in lines:
        if re.search(r"\bQty\b.*\bItem\b", line, re.I):
            in_items = True
            continue

        if not in_items:
            continue

        if stop_words.search(line):
            break

        m = pattern.match(line)
        if m:
            qty = int(m.group(1))
            description = m.group(2).strip(" -:")
            price = money_to_float(m.group(3))

            items.append(
                {
                    "quantity": qty,
                    "item": description,
                    "price": price,
                    "line_total": round(qty * price, 2) if price is not None else None,
                }
            )

    return items


# ---------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------

def read_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()

    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
        return [Image.open(io.BytesIO(data)).convert("RGB")]

    if suffix == ".pdf":
        return pdf_to_images(data)

    raise ValueError(f"Unsupported file type: {suffix}")


def expand_zip(uploaded_file):
    """Return [(filename, bytes), ...] for images/PDFs contained in a ZIP."""
    output = []
    with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue

            suffix = Path(name).suffix.lower()
            if suffix in {
                ".jpg", ".jpeg", ".png", ".webp",
                ".bmp", ".tif", ".tiff", ".pdf"
            }:
                output.append((name, z.read(name)))

    return output


# ---------------------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    current_year = datetime.now().year
    selected_year = st.number_input(
        "Year-to-date year",
        min_value=2000,
        max_value=2100,
        value=current_year,
        step=1,
    )

    show_raw = st.checkbox("Show OCR text", value=False)
    show_items = st.checkbox("Show extracted items", value=True)

    st.info(
        "For best OCR accuracy, use clear photos/scans of the complete receipt. "
        "The parser is tuned to the restaurant receipt layout shown in your sample."
    )

uploaded = st.file_uploader(
    "Upload invoices",
    type=[
        "jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff",
        "pdf", "zip"
    ],
    accept_multiple_files=True,
)

if not uploaded:
    st.markdown(
        """
        ### What this app does

        Upload all your restaurant invoices together. The application will:

        1. OCR each invoice.
        2. Extract the invoice date, order/check numbers, customer, items,
           subtotal, VAT and total.
        3. Compile everything into one table.
        4. Calculate year-to-date subtotal, VAT and total.
        5. Allow the consolidated data to be downloaded as CSV/Excel.

        You can upload individual receipt photos, PDFs, or a ZIP containing receipts.
        """
    )
    st.stop()


files_to_process = []

for uploaded_file in uploaded:
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".zip":
        try:
            for name, data in expand_zip(uploaded_file):
                files_to_process.append((name, data))
        except Exception as exc:
            st.error(f"Could not read ZIP file {uploaded_file.name}: {exc}")
    else:
        files_to_process.append((uploaded_file.name, uploaded_file.getvalue()))


invoice_records = []
item_records = []

progress = st.progress(0)
status = st.empty()

for index, (filename, data) in enumerate(files_to_process, start=1):
    status.write(f"Processing {index}/{len(files_to_process)}: {filename}")

    try:
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            images = pdf_to_images(data)
        else:
            images = [Image.open(io.BytesIO(data)).convert("RGB")]

        # One PDF can contain multiple invoice pages.
        for page_number, image in enumerate(images, start=1):
            page_name = (
                filename if len(images) == 1
                else f"{filename} — page {page_number}"
            )

            text = ocr_image(image)
            record = parse_invoice(text, page_name)

            # If a field is missing from OCR, calculate it where possible.
            if record["total"] is None and record["subtotal"] is not None and record["vat"] is not None:
                record["total"] = round(record["subtotal"] + record["vat"], 2)

            if record["subtotal"] is None and record["total"] is not None and record["vat"] is not None:
                record["subtotal"] = round(record["total"] - record["vat"], 2)

            invoice_records.append(record)

            for item in extract_items(text):
                item["filename"] = page_name
                item_records.append(item)

            if show_raw:
                with st.expander(f"OCR: {page_name}"):
                    st.text(text)

    except Exception as exc:
        st.error(f"Could not process {filename}: {exc}")

    progress.progress(index / max(len(files_to_process), 1))

status.empty()

if not invoice_records:
    st.warning("No invoices could be extracted.")
    st.stop()


df = pd.DataFrame(invoice_records)

# Ensure date column is a real datetime column.
df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")

# Sort chronologically.
df = df.sort_values(["invoice_date", "filename"], na_position="last").reset_index(drop=True)

# ---------------------------------------------------------------------
# Year-to-date calculation
# ---------------------------------------------------------------------

today = pd.Timestamp.today().normalize()
start_of_year = pd.Timestamp(year=int(selected_year), month=1, day=1)

if int(selected_year) == today.year:
    ytd_end = today
else:
    ytd_end = pd.Timestamp(year=int(selected_year), month=12, day=31)

ytd_mask = (
    df["invoice_date"].notna()
    & (df["invoice_date"] >= start_of_year)
    & (df["invoice_date"] <= ytd_end)
)

ytd = df.loc[ytd_mask].copy()

invoice_count = len(ytd)
subtotal_ytd = pd.to_numeric(ytd["subtotal"], errors="coerce").sum()
vat_ytd = pd.to_numeric(ytd["vat"], errors="coerce").sum()
total_ytd = pd.to_numeric(ytd["total"], errors="coerce").sum()

# ---------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------

st.subheader(f"Year-to-date summary — {selected_year}")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Invoices", f"{invoice_count:,}")
c2.metric("Subtotal", f"{subtotal_ytd:,.2f}")
c3.metric("VAT", f"{vat_ytd:,.2f}")
c4.metric("Total", f"{total_ytd:,.2f}")

if invoice_count:
    st.caption(
        f"YTD period: {start_of_year.date()} through {ytd_end.date()}. "
        "Only invoices whose extracted invoice date falls within this period are included."
    )
else:
    st.warning(
        "No invoice dates were successfully extracted for the selected YTD period. "
        "Check the OCR text below or use clearer receipt images."
    )


# ---------------------------------------------------------------------
# Invoice table
# ---------------------------------------------------------------------

st.subheader("All extracted invoices")

display_columns = [
    "invoice_date",
    "filename",
    "order_number",
    "check_number",
    "customer",
    "service_type",
    "subtotal",
    "vat",
    "total",
    "payment",
    "products_count",
]

available_columns = [c for c in display_columns if c in df.columns]

st.dataframe(
    df[available_columns],
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------

if show_items and item_records:
    st.subheader("Extracted items")
    items_df = pd.DataFrame(item_records)
    st.dataframe(items_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Missing / suspicious fields
# ---------------------------------------------------------------------

st.subheader("OCR quality checks")

checks = []

for _, row in df.iterrows():
    problems = []

    if pd.isna(row["invoice_date"]):
        problems.append("date missing")

    if pd.isna(row["subtotal"]):
        problems.append("subtotal missing")

    if pd.isna(row["total"]):
        problems.append("total missing")

    if pd.notna(row["subtotal"]) and pd.notna(row["vat"]) and pd.notna(row["total"]):
        expected = round(float(row["subtotal"]) + float(row["vat"]), 2)
        if abs(expected - float(row["total"])) > 0.02:
            problems.append(
                f"subtotal + VAT ≠ total ({expected:.2f} vs {float(row['total']):.2f})"
            )

    if problems:
        checks.append(
            {
                "filename": row["filename"],
                "issues": "; ".join(problems),
            }
        )

if checks:
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
else:
    st.success("No obvious date/subtotal/VAT/total inconsistencies were detected.")


# ---------------------------------------------------------------------
# Monthly YTD summary
# ---------------------------------------------------------------------

if not ytd.empty:
    st.subheader("Monthly YTD summary")

    monthly = (
        ytd.assign(month=ytd["invoice_date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            invoices=("filename", "count"),
            subtotal=("subtotal", "sum"),
            vat=("vat", "sum"),
            total=("total", "sum"),
        )
    )

    st.dataframe(monthly, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------

st.subheader("Export")

csv_data = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "Download all invoices as CSV",
    data=csv_data,
    file_name="restaurant_invoices_all.csv",
    mime="text/csv",
)

if item_records:
    items_csv = pd.DataFrame(item_records).to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download extracted items as CSV",
        data=items_csv,
        file_name="restaurant_invoice_items.csv",
        mime="text/csv",
    )

# Excel export is optional and requires openpyxl.
try:
    import openpyxl

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Invoices", index=False)

        if item_records:
            pd.DataFrame(item_records).to_excel(
                writer,
                sheet_name="Items",
                index=False,
            )

        if not ytd.empty:
            monthly.to_excel(
                writer,
                sheet_name="Monthly YTD",
                index=False,
            )

        pd.DataFrame(
            [
                ["YTD year", int(selected_year)],
                ["Invoice count", invoice_count],
                ["Subtotal", round(subtotal_ytd, 2)],
                ["VAT", round(vat_ytd, 2)],
                ["Total", round(total_ytd, 2)],
            ],
            columns=["Metric", "Value"],
        ).to_excel(writer, sheet_name="YTD Summary", index=False)

    st.download_button(
        "Download Excel workbook",
        data=excel_buffer.getvalue(),
        file_name="restaurant_invoice_consolidated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

except Exception:
    st.info("Install openpyxl to enable Excel export.")


# ---------------------------------------------------------------------
# Important note
# ---------------------------------------------------------------------

st.caption(
    "OCR results should be reviewed against the original receipts before using "
    "the figures for accounting, tax, reimbursement, or legal purposes."
)
