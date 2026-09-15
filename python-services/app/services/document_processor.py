from pathlib import Path
from .extractor.docling_extractor import DoclingExtractor
from .extractor.image_extractor import ImageExtractor


class DocumentProcessor:
    def __init__(self, file_path, user_id, document_id):
        self.root_path = Path(
            "/home/bardia/Desktop/graph-rag/backend/storage/app/public"
        )
        self.file_path = self.root_path / file_path
        self.user_id = user_id
        self.document_id = document_id

        # ✅ document_id پاس داده می‌شود
        self.extractor = DoclingExtractor(
            self.file_path,
            document_id=document_id,
        )

    def extract_document(self):
        """PDF را با docling پردازش می‌کند و DoclingDocument برمی‌گرداند."""
        return self.extractor.load_document()

    def extract_standard(self):
        """خروجی نرمال‌شده استاندارد (NormalizedDocument)"""
        return self.extractor.normalize()

    def extract_image(self):
        """تصاویر را از PDF استخراج می‌کند."""
        ImageExtractor(self.file_path).extract_image()