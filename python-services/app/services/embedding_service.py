from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = "cpu",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device or "cpu"
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name,
            device=self.device,
        )

        self.dimensions = self.model.get_embedding_dimension()

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> np.ndarray:

        if not texts:
            return np.empty((0, self.dimensions))

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return embeddings

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        text_field: str = "content",
        show_progress: bool = False,
    ) -> List[Dict[str, Any]]:

        if not chunks:
            return []

        texts = []

        for index, chunk in enumerate(chunks):
            text = chunk.get(text_field, "")

            if not isinstance(text, str):
                text = str(text)

            text = text.strip()

            if not text:
                raise ValueError(
                    f"Chunk at index {index} has empty '{text_field}'"
                )

            texts.append(text)

        embeddings = self.embed_texts(
            texts,
            show_progress=show_progress,
        )

        embedded_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            new_chunk = dict(chunk)

            new_chunk["embedding"] = embedding.tolist()

            embedded_chunks.append(new_chunk)

        return embedded_chunks

    def get_info(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "device": self.device,
            "batch_size": self.batch_size,
        }





    def embed_question(self, question: str) -> List[float]:

        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        embedding = self.embed_texts([question])[0]

        return embedding.tolist()
    