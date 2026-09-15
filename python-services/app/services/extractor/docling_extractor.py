from docling.document_converter import DocumentConverter
from ..docling_inspector import DoclingInspector
from ..docling_normalizer import DoclingNormalizer


class DoclingExtractor:
    def __init__(self, file_path, document_id=0):
        self.file_path = file_path
        self.document_id = document_id
        self.doc = None

    def load_document(self):
        """PDF را با docling پردازش می‌کند."""
        converter = DocumentConverter()
        result = converter.convert(str(self.file_path))
        self.doc = result.document
        return self.doc

    def inspect(self) -> dict:
        """خروجی خام از DoclingInspector (برای debug)"""
        if self.doc is None:
            self.load_document()
        return DoclingInspector(self.doc).inspect()

    def pretty_print(self) -> str:
        """خروجی خوانا برای debug"""
        if self.doc is None:
            self.load_document()
        return DoclingInspector(self.doc).pretty_print()

    def normalize(self) -> dict:
        """خروجی استاندارد (NormalizedDocument)"""
        if self.doc is None:
            self.load_document()

        raw = DoclingInspector(self.doc).inspect()
        normalizer = DoclingNormalizer(
            raw["elements"],
            document_id=self.document_id,
        )
        normalized = normalizer.normalize()
        normalized["title"] = raw["title"]
        return normalized