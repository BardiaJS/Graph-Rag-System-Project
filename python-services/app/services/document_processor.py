from pathlib import Path

from .extractor.docling_extractor import DoclingExtractor
from .extractor.image_extractor import ImageExtractor
from .semantic_chunker import SemanticChunker
from .embedding_service import EmbeddingService


class DocumentProcessor:
    def __init__(self, file_path, user_id, document_id):
        self.root_path = Path(
            "/home/bardia/Desktop/graph-rag/backend/storage/app/public"
        )
        self.file_path = self.root_path / file_path
        self.user_id = user_id
        self.document_id = document_id

        self.extractor = DoclingExtractor(
            self.file_path,
            document_id=document_id,
        )

    # ============================================================
    # Document Extraction
    # ============================================================
    def extract_document(self):
        """PDF را با docling پردازش می‌کند."""
        return self.extractor.load_document()

    def extract_standard(self):
        """خروجی نرمال‌شده (NormalizedDocument)"""
        return self.extractor.normalize()

    def extract_image(self):
        """تصاویر را از PDF استخراج می‌کند."""
        ImageExtractor(self.file_path).extract_image()

    # ============================================================
    # Chunking
    # ============================================================
    def chunk_document(
        self,
        normalized_doc: dict = None,
        max_tokens: int = 512,
        min_tokens: int = 100,
        similarity_threshold: float = 0.70,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_overlap: int = 50,
    ) -> dict:
        """
        NormalizedDocument را به chunk های معنادار تبدیل می‌کند.
        """

        if normalized_doc is None:
            normalized_doc = self.extract_standard()

        chunker = SemanticChunker(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            similarity_threshold=similarity_threshold,
            model_name=embedding_model,
        )

        chunks = chunker.chunk(
            normalized_doc["elements"]
        )

        return {
            "document_id": self.document_id,
            "title": normalized_doc.get("title"),
            "num_chunks": len(chunks),
            "stats": chunker.get_stats(chunks),
            "chunks": chunks,
        }

