from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingService:

    def __init__(self, pages, document_id):
        self.pages = pages
        self.document_id = document_id

    def chunk(self):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = []
        chunk_index = 0

        for page in self.pages:

            page_chunks = splitter.split_text(page["text"])

            for chunk in page_chunks:

                chunks.append({
                    "text": chunk,
                    "document_id": self.document_id,
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index
                })

                chunk_index += 1

        return chunks