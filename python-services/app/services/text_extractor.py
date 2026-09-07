from langchain_community.document_loaders import PyMuPDFLoader
import fitz


class PdfTextExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def extract(self):
        loader = PyMuPDFLoader(self.file_path)
        data = loader.load()

        doc = fitz.open(self.file_path)

        metadata = doc.metadata
        page_count = len(doc)

        pages = []

        for page in data:
            pages.append({
                "page_number": page.metadata["page"] + 1,
                "text": page.page_content
            })

        doc.close()
            
        return {
            "page_count": page_count,
            "metadata": metadata,
            "pages": pages
        }