from pathlib import Path
from .extractor.docling_extractor import DoclingExtractor
from .extractor.image_extractor import ImageExtractor

class DocumentProcessor:
    def __init__(self, file_path, user_id, document_id):
        self.root_path = Path("/home/bardia/Desktop/graph-rag/backend/storage/app/public")
        self.file_path = self.root_path / file_path
        self.user_id = user_id
        self.document_id = document_id

    def extract_document(self):
        extractor = DoclingExtractor(self.file_path)
        return extractor.load_document()   # ← DoclingDocument

    def extract_image(self):
        image_extracted = ImageExtractor(self.file_path)
        image_extracted.extract_image()