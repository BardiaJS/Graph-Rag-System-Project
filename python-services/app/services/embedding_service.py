from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingService:
    """سرویس تولید embedding و محاسبه شباهت بین چانک‌ها."""

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, device: str = "cpu"):
        self.model = SentenceTransformer(
            self.MODEL_NAME,
            device=device,
        )

    def embed_chunks(self, chunks: list[dict]):
        """لیست چانک‌ها را به بردار تبدیل می‌کند."""
        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        return embeddings

    def compute_similarity(self, embeddings):
        """ماتریس شباهت کسینوسی."""
        return cosine_similarity(embeddings)

    def most_similar(self, embeddings, target_index: int = 0):
        """ایندکس شبیه‌ترین چانک به چانک هدف (به جز خودش)."""
        sim = self.compute_similarity(embeddings)
        similarities = sim[target_index].copy()
        similarities[target_index] = -1
        return int(similarities.argmax()), similarities