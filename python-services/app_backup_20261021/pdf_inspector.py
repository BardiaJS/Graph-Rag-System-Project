"""
Decide whether a PDF needs OCR by checking which pages carry a real text layer.

Why not just always enable OCR? Docling's OCR also runs over figure bitmaps in ordinary
digital papers: slower, and it injects stray text read out of charts into your chunks.
So OCR is switched on only when the document actually lacks text.

    digital : most pages have a text layer          -> no OCR
    mixed   : a sizeable share of pages lack text   -> OCR (keeps existing text on the rest)
    scanned : (almost) no page has a text layer     -> full-page OCR

pypdfium2 is already installed as a Docling dependency.
"""
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# A page with fewer visible characters than this counts as "no text layer"
# (a scan with only a printed page number would otherwise look digital).
MIN_CHARS_PER_PAGE = int(os.getenv("OCR_MIN_CHARS_PER_PAGE", 50))
# Share of text-less pages that switches OCR on. 0.3 means a digital paper with one or
# two full-page figures is NOT sent through OCR, but a half-scanned one is.
OCR_PAGE_FRACTION = float(os.getenv("OCR_PAGE_FRACTION", 0.3))
SCANNED_FRACTION = 0.9
MAX_SAMPLED_PAGES = 100  # bound the cost on very long documents


def classify(
    char_counts: List[int],
    min_chars: int = MIN_CHARS_PER_PAGE,
    fraction: float = OCR_PAGE_FRACTION,
) -> Dict[str, Any]:
    """Pure function over per-page character counts."""
    total = len(char_counts)
    if total == 0:
        return {"kind": "unknown", "needs_ocr": False,
                "pages_checked": 0, "pages_without_text": 0}

    without_text = sum(1 for c in char_counts if c < min_chars)
    share = without_text / total

    if share >= SCANNED_FRACTION:
        kind = "scanned"
    elif share >= fraction:
        kind = "mixed"
    else:
        kind = "digital"

    return {
        "kind": kind,
        "needs_ocr": kind in ("scanned", "mixed"),
        "pages_checked": total,
        "pages_without_text": without_text,
    }


def inspect_pdf(path: str) -> Dict[str, Any]:
    """Never raises: on any problem (encrypted file, missing library) it reports
    kind='unknown' and needs_ocr=False, i.e. behaves exactly as before this feature."""
    try:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(path)
        try:
            n = len(pdf)
            step = max(1, n // MAX_SAMPLED_PAGES)
            counts: List[int] = []
            for i in range(0, n, step):
                page = pdf[i]
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_range() or ""
                    counts.append(len("".join(text.split())))  # visible chars only
                finally:
                    textpage.close()
                    page.close()
        finally:
            pdf.close()

        info = classify(counts)
        info["pages_total"] = n
        return info
    except Exception as e:
        logger.warning(f"Could not inspect PDF text layer ({path}): {e}")
        return {"kind": "unknown", "needs_ocr": False,
                "pages_checked": 0, "pages_without_text": 0, "error": str(e)}