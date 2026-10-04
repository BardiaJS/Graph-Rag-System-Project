from app.services.docling_service import DoclingService


class DocumentProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.service = DoclingService(file_path)

    def docling_service(self):
        return self.service.docling_funtion()

    def chunking_service(self):
        return self.service.chunking_service()