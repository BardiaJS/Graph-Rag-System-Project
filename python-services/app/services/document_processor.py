from .text_extractor import PdfTextExtractor
from .image_extractor import ImageExtractor
from .chunking_service import ChunkingService
from pathlib import Path
import fitz

class DocumentProcessor:
    def __init__(self, file_path , user_id , document_id):
        self.root_path = Path('/home/bardia/Desktop/graph-rag/backend/storage/app/public')
        self.file_path = self.root_path / file_path
        self.document_id = document_id
        self.text_extractor = PdfTextExtractor(self.file_path)
        self.image_extractor = ImageExtractor(self.file_path , user_id , document_id)

    def text_extraction(self):
        extracted_data = self.text_extractor.extract()

        return extracted_data

    def image_extraction(self):
        extracted_image = self.image_extractor.image_extract()

        return extracted_image

    def chunk_service(self, pages):
        chunk_extractor = ChunkingService(
            pages,
            self.document_id
        )

        return chunk_extractor.chunk()


    


        




        