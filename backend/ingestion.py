# backend/ingestion.py
import io
import os
from typing import Dict, Any, List
import shutil
import pdfplumber
from PIL import Image
import pytesseract

# ---------- TESSERACT SETUP (Windows-friendly) ----------

# Try to find tesseract.exe automatically
_tess_cmd = shutil.which("tesseract")

if _tess_cmd is None:
    # Common Windows install locations
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            _tess_cmd = path
            break

if _tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tess_cmd
    print(f"[OCR] Using Tesseract at: {_tess_cmd}")
else:
    # We won't crash here; _process_image will handle the missing binary
    print("[OCR] WARNING: tesseract.exe not found. Image OCR will be unavailable.")


def _assess_legibility(texts: List[str]) -> bool:
    """Very simple legibility heuristic.
    If the total extracted text length across all pages is above a small threshold,
    we mark the document as legible. Otherwise we ask the user to check manually.
    """
    total_chars = sum(len((t or "").strip()) for t in texts)
    # Tunable threshold; small to avoid false negatives on short docs
    return total_chars >= 30


def _process_pdf(file_bytes: bytes) -> Dict[str, Any]:
    """Extract text & image counts from a PDF file.

    Returns a normalized doc_info dict without the filename (added later):
        {
            "num_pages": int,
            "num_images": int,
            "legible": bool,
            "pages": [
                {"page_num": int, "text": str},
                ...
            ],
        }
    """
    pages: List[Dict[str, Any]] = []
    page_texts: List[str] = []
    num_images = 0

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            page_texts.append(text)

            # pdfplumber exposes image metadata per page
            num_images += len(page.images)

            pages.append(
                {
                    "page_num": i,
                    "text": text,
                }
            )

    num_pages = len(pages)
    legible = _assess_legibility(page_texts)

    return {
        "num_pages": num_pages,
        "num_images": num_images,
        "legible": legible,
        "pages": pages,
    }


def _process_image(file_bytes: bytes) -> Dict[str, Any]:
    """Handle standalone image files (PNG / JPG / JPEG).

    We treat each uploaded image as a single-page document with:
        - num_pages = 1
        - num_images = 1
        - OCR text extracted via Tesseract
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # OCR using Tesseract
    ocr_text = pytesseract.image_to_string(image) or ""

    pages = [
        {
            "page_num": 1,
            "text": ocr_text,
        }
    ]

    legible = _assess_legibility([ocr_text])

    return {
        "num_pages": 1,
        "num_images": 1,
        "legible": legible,
        "pages": pages,
    }


def process_file(uploaded_file) -> Dict[str, Any]:
    """Accepts a Streamlit UploadedFile and returns a normalized doc_info dict.

    This is the single entrypoint used by the Streamlit app. It supports:
      - PDFs (multi-page, text + embedded images)
      - Standalone images (PNG / JPG / JPEG) via OCR
    """
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        info = _process_pdf(file_bytes)
    elif any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg")):
        info = _process_image(file_bytes)
    else:
        # Fallback: try image path first, then treat as zero-page doc
        try:
            info = _process_image(file_bytes)
        except Exception:
            info = {
                "num_pages": 0,
                "num_images": 0,
                "legible": False,
                "pages": [],
            }

    info["filename"] = uploaded_file.name
    return info
