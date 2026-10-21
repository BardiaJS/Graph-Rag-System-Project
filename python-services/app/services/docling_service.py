import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

from app.services.pdf_inspector import inspect_pdf

logger = logging.getLogger(__name__)

# Render scale for cropped figure images (2.0 ~ 144 dpi). Higher = sharper, more memory.
FIGURE_IMAGES_SCALE = float(os.getenv("DOCLING_IMAGES_SCALE", 2.0))
# Ignore pictures smaller than this (pixels, at the scale above): logos, icons, rules.
FIGURE_MIN_PX = int(os.getenv("FIGURE_MIN_PX", 150))

# OCR: auto = only when the PDF lacks a text layer (see pdf_inspector); on / off force it.
OCR_MODE = os.getenv("DOCLING_OCR", "auto").lower()
OCR_ENGINE = os.getenv("DOCLING_OCR_ENGINE", "easyocr").lower()  # easyocr | tesseract
# Language codes are engine-specific: easyocr "en,fa"  /  tesseract "eng,fas"
OCR_LANGS = os.getenv("DOCLING_OCR_LANGS", "")


def _build_ocr_options(force_full_page: bool):
    """Imported lazily so a missing OCR extra never breaks normal (digital) processing."""
    if OCR_ENGINE == "tesseract":
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions

        langs = [l.strip() for l in (OCR_LANGS or "eng,fas").split(",") if l.strip()]
        return TesseractCliOcrOptions(lang=langs, force_full_page_ocr=force_full_page)

    from docling.datamodel.pipeline_options import EasyOcrOptions

    langs = [l.strip() for l in (OCR_LANGS or "en,fa").split(",") if l.strip()]
    return EasyOcrOptions(lang=langs, force_full_page_ocr=force_full_page)


class DoclingService:
    _tokenizer = None

    def __init__(self, file_path):
        self.file_path = file_path
        self._doc = None  # cached per instance
        self._ocr_plan = None

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2"
                ),
                max_tokens=256,
            )
        return cls._tokenizer

    def plan_ocr(self) -> Dict[str, Any]:
        """Decide (once) whether this PDF needs OCR. Cheap: it only reads the text layer.
        Returns e.g. {kind, needs_ocr, pages_checked, pages_without_text, mode, use_ocr}."""
        if self._ocr_plan is None:
            info = inspect_pdf(self.file_path)
            if OCR_MODE == "on":
                use = True
            elif OCR_MODE == "off":
                use = False
            else:
                use = bool(info["needs_ocr"])
            self._ocr_plan = {**info, "mode": OCR_MODE, "use_ocr": use,
                              "engine": OCR_ENGINE if use else None}
            logger.info(f"OCR plan for {self.file_path}: {self._ocr_plan}")
        return self._ocr_plan

    def _convert(self):
        """Convert the PDF to a docling document once; cached."""
        if self._doc is not None:
            return self._doc

        pipeline_options = PdfPipelineOptions()
        plan = self.plan_ocr()
        pipeline_options.do_ocr = plan["use_ocr"]
        if plan["use_ocr"]:
            # Pure scans: OCR the whole page. Mixed documents: keep the existing text
            # layer where there is one and OCR only the bitmap-only pages.
            pipeline_options.ocr_options = _build_ocr_options(
                force_full_page=plan["kind"] == "scanned"
            )
        pipeline_options.do_table_structure = False
        # Needed so PictureItem.get_image() can return the cropped figure.
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = FIGURE_IMAGES_SCALE
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=2,
            device=AcceleratorDevice.CPU,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self._doc = converter.convert(self.file_path).document
        return self._doc

    def docling_funtion(self):
        """markdown"""
        doc = self._convert()
        return doc.export_to_markdown()

    def chunking_service(self):
        """chunk + metadata (no embedding)"""
        doc = self._convert()

        chunker = HybridChunker(
            tokenizer=self._get_tokenizer(),
            merge_peers=True,
            repeat_table_header=True,
        )

        chunks = list(chunker.chunk(dl_doc=doc))

        results = []
        for chunk in chunks:
            enriched_text = chunker.contextualize(chunk=chunk)

            # page number from doc_items
            page = None
            if chunk.meta and chunk.meta.doc_items:
                first_item = chunk.meta.doc_items[0]
                prov = getattr(first_item, "prov", None)
                if prov:
                    page = getattr(prov[0], "page_no", None)

            results.append({
                "text": chunk.text,
                "enriched_text": enriched_text,
                "headings": chunk.meta.headings if chunk.meta else [],
                "page": page,
            })

        return results

    def extract_figures(self, output_dir) -> List[Dict[str, Any]]:
        """Save every meaningful picture as PNG and return its metadata.

        Reuses the cached conversion, so call it after (or before) chunking with no extra
        parsing cost. Each returned dict: figure_index, caption, page, bbox [l,t,r,b],
        image_file, width, height. One broken picture never aborts the rest."""
        doc = self._convert()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        figures: List[Dict[str, Any]] = []
        for pic in doc.pictures:
            try:
                img = pic.get_image(doc)
                if img is None:
                    continue
                width, height = img.size
                if width < FIGURE_MIN_PX or height < FIGURE_MIN_PX:
                    continue

                index = len(figures)
                filename = f"fig_{index}.png"
                img.convert("RGB").save(out / filename, format="PNG")

                caption = ""
                try:
                    caption = (pic.caption_text(doc) or "").strip()
                except Exception:
                    pass

                prov = pic.prov[0] if getattr(pic, "prov", None) else None
                page = getattr(prov, "page_no", None)
                bbox = None
                box = getattr(prov, "bbox", None)
                if box is not None:
                    bbox = [float(box.l), float(box.t), float(box.r), float(box.b)]

                figures.append({
                    "figure_index": index,
                    "caption": caption,
                    "page": page,
                    "bbox": bbox,
                    "image_file": filename,
                    "width": width,
                    "height": height,
                })
            except Exception as e:
                logger.warning(f"Skipping a picture that could not be extracted: {e}")

        logger.info(f"Extracted {len(figures)} figures to {out}")
        return figures
