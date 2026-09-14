from pathlib import Path
from .structure_detector import StructureDetector
from .extractor.text_extractor import TextExtractor
from .extractor.image_extractor import ImageExtractor



class DocumentProcessor:
    def __init__(self, file_path, user_id, document_id):
        self.root_path = Path(
            "/home/bardia/Desktop/graph-rag/backend/storage/app/public"
        )

        self.file_path = self.root_path / file_path
        self.user_id = user_id
        self.document_id = document_id

    def detect_struture(self):
        StructureDetector.detect_structure(self.file_path)

    def extract_text(self):
        text_extrected = TextExtractor(self.file_path)
        pages = text_extrected.load_pages()
        text_extrected.count_page_number()
        text_extrected.get_metadata()
        # text_extrected.extract_blocks()
        # text_extrected.extract_words()




    def extract_image(self):
        image_extracted = ImageExtractor(self.file_path)
        image_extracted.extract_image()
        



    # def text_extraction(self):
    #     return self.text_extractor.extract()

    # def image_extraction(self):
    #     return self.image_extractor.image_extract()

    # def structure_extraction(self):
    #     extractor = DocumentStructureExtractor(
    #         self.file_path,
    #         self.document_id,
    #     )
    #     return extractor.extract()

    # def chunk_service(self, pages):
    #     chunk_extractor = ChunkingService(pages, self.document_id)
    #     return chunk_extractor.chunk()

    # def embedding(self, chunks):
    #     """چانک‌ها را embed می‌کند و نتیجه را برمی‌گرداند."""
    #     return self.embedding_service.embed_chunks(chunks)