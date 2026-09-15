from docling.document_converter import DocumentConverter

class DoclingExtractor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_document(self):
        converter = DocumentConverter()
        result = converter.convert(str(self.file_path))
        return result.document   # ← DoclingDocument

    def to_markdown(self, doc):
        return doc.export_to_markdown()

    def to_dict(self, doc):
        return doc.export_to_dict()