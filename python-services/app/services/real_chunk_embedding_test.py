import sys

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.text_extractor import PdfTextExtractor
from app.services.chunking_service import ChunkingService


class ChunkEmbeddingTester:
    """تست استخراج، تکه‌بندی و embed کردن چانک‌های یک PDF."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, pdf_path: str, document_id: int = 1):
        self.pdf_path = pdf_path
        self.document_id = document_id

        self.chunks: list[dict] = []
        self.embeddings = None
        self.similarity_matrix = None

        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device="cpu",
        )

    # --------------------------------------------------
    # 1. PDF -> Pages
    # --------------------------------------------------
    def extract_pages(self) -> list:
        extractor = PdfTextExtractor(self.pdf_path)
        return extractor.extract()

    # --------------------------------------------------
    # 2. Pages -> Chunks
    # --------------------------------------------------
    def build_chunks(self, pages) -> list[dict]:
        chunking_service = ChunkingService(pages, self.document_id)
        self.chunks = chunking_service.chunk()
        return self.chunks

    def print_chunk_info(self) -> None:
        print("\n========== CHUNK INFO ==========\n")
        print("TOTAL CHUNKS:", len(self.chunks))

    # --------------------------------------------------
    # 3. Chunks -> Embeddings
    # --------------------------------------------------
    def build_embeddings(self):
        texts = [chunk["text"] for chunk in self.chunks]
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return self.embeddings

    # --------------------------------------------------
    # 4. Validate embeddings
    # --------------------------------------------------
    def print_embedding_info(self) -> None:
        print("\n========== EMBEDDING INFO ==========\n")
        print("TOTAL EMBEDDINGS:", len(self.embeddings))
        print("VECTOR DIMENSION:", len(self.embeddings[0]))

        for chunk, embedding in zip(self.chunks[:5], self.embeddings[:5]):
            print("=" * 70)
            print("CHUNK INDEX:", chunk["chunk_index"])
            print("PAGE:", chunk["page_number"])
            print("TEXT:", chunk["text"][:300].replace("\n", " "))
            print("DIMENSION:", len(embedding))
            print(
                "NORM:",
                round(float((embedding ** 2).sum() ** 0.5), 4),
            )

    # --------------------------------------------------
    # 5. Cosine similarity between real chunks
    # --------------------------------------------------
    def compute_similarity(self):
        print("\n========== REAL CHUNK SIMILARITY ==========\n")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        return self.similarity_matrix

    def print_pairwise_similarity(self, limit: int = 5) -> None:
        limit = min(limit, len(self.chunks))
        for i in range(limit):
            for j in range(i + 1, limit):
                print(
                    f"Chunk {i} <-> Chunk {j}: "
                    f"{self.similarity_matrix[i][j]:.4f}"
                )

    # --------------------------------------------------
    # 6. Find most similar chunk to target chunk
    # --------------------------------------------------
    def find_most_similar(self, target_index: int = 0) -> None:
        if len(self.chunks) <= 1:
            return

        similarities = self.similarity_matrix[target_index].copy()
        similarities[target_index] = -1  # خودش را حذف کن

        most_similar_index = int(similarities.argmax())

        print("\n========== MOST SIMILAR CHUNK ==========\n")
        print("TARGET CHUNK:", target_index)
        print("TARGET PAGE:", self.chunks[target_index]["page_number"])
        print(
            "\nTARGET TEXT:\n",
            self.chunks[target_index]["text"][:500],
        )

        print("\nMOST SIMILAR CHUNK:", most_similar_index)
        print(
            "MOST SIMILAR PAGE:",
            self.chunks[most_similar_index]["page_number"],
        )
        print(
            "SIMILARITY:",
            round(float(similarities[most_similar_index]), 4),
        )
        print(
            "\nMOST SIMILAR TEXT:\n",
            self.chunks[most_similar_index]["text"][:500],
        )

    # --------------------------------------------------
    # Run full pipeline
    # --------------------------------------------------
    def run(self) -> None:
        pages = self.extract_pages()
        self.build_chunks(pages)
        self.print_chunk_info()

        self.build_embeddings()
        self.print_embedding_info()

        self.compute_similarity()
        self.print_pairwise_similarity(limit=5)

        self.find_most_similar(target_index=0)


# --------------------------------------------------
# CLI entry point
# --------------------------------------------------
def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python app/services/real_chunk_embedding_test.py "
            "<pdf_path> [document_id]"
        )
        sys.exit(1)

    pdf_path = sys.argv[1]
    document_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    tester = ChunkEmbeddingTester(pdf_path, document_id)
    tester.run()


if __name__ == "__main__":
    main()