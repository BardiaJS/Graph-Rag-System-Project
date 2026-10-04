from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer


class DoclingService:
    _tokenizer = None

    def __init__(self, file_path):
        self.file_path = file_path
        self._doc = None  # ← کش داخل نمونه

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
        """تبدیل PDF به docling document — یه بار، کش می‌شه"""
        if self._doc is not None:
            return self._doc

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
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
        """chunk + metadata (بدون embedding)"""
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

            # page number از doc_items
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