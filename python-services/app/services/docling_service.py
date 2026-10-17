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

logger = logging.getLogger(__name__)

# Render scale for cropped figure images (2.0 ~ 144 dpi). Higher = sharper, more memory.
FIGURE_IMAGES_SCALE = float(os.getenv("DOCLING_IMAGES_SCALE", 2.0))
# Ignore pictures smaller than this (pixels, at the scale above): logos, icons, rules.
FIGURE_MIN_PX = int(os.getenv("FIGURE_MIN_PX", 150))


class DoclingService:
    _tokenizer = None

    def __init__(self, file_path):
        self.file_path = file_path
        self._doc = None  # cached per instance

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

    def _convert(self):
        """Convert the PDF to a docling document once; cached."""
        if self._doc is not None:
            return self._doc

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
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