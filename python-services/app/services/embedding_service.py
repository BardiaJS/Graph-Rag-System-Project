import hashlib
import logging
import uuid

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding generation + Qdrant vector storage/search.

    Model:
        sentence-transformers/all-MiniLM-L6-v2

    Vector size:
        384
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    COLLECTION_NAME = "documents"

    VECTOR_SIZE = 384

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        model: Optional[SentenceTransformer] = None,
    ):
        logger.info("Initializing EmbeddingService...")

        # ==================================================
        # Embedding model
        # ==================================================

        if model is not None:
            logger.info("Using shared SentenceTransformer model")
            self.model = model
        else:
            logger.info(
                "Loading SentenceTransformer model: %s",
                self.MODEL_NAME,
            )

            self.model = SentenceTransformer(
                self.MODEL_NAME,
                device="cpu",
            )

        # ==================================================
        # Qdrant
        # ==================================================

        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
        )

        self.collection_name = self.COLLECTION_NAME
        self.vector_size = self.VECTOR_SIZE

        self._create_collection()
        self._ensure_payload_indexes()

        logger.info(
            "EmbeddingService initialized successfully"
        )

    # ==========================================================
    # Collection
    # ==========================================================

    def _create_collection(self) -> None:
        """
        Create Qdrant collection if it does not exist.
        """

        try:
            collections = self.client.get_collections()

            collection_names = {
                collection.name
                for collection in collections.collections
            }

            if self.collection_name not in collection_names:

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )

                logger.info(
                    "Qdrant collection '%s' created",
                    self.collection_name,
                )

            else:

                logger.info(
                    "Qdrant collection '%s' already exists",
                    self.collection_name,
                )

        except Exception:

            logger.exception(
                "Failed to create/check Qdrant collection"
            )

            raise

    # ==========================================================
    # Payload indexes
    # ==========================================================

    def _ensure_payload_indexes(self) -> None:
        """
        Ensure frequently filtered payload fields have indexes.
        """

        try:

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema="integer",
                wait=True,
            )

            logger.info(
                "Qdrant payload index for 'doc_id' is ready"
            )

        except Exception:

            logger.exception(
                "Failed to create payload index for doc_id"
            )

    # ==========================================================
    # Embedding
    # ==========================================================

    def generate_embedding(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate embedding for a single text.
        """

        if not text or not text.strip():
            return [0.0] * self.vector_size

        try:

            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            result = embedding.tolist()

            if len(result) != self.vector_size:
                raise ValueError(
                    f"Invalid embedding dimension: "
                    f"expected={self.vector_size}, "
                    f"actual={len(result)}"
                )

            return result

        except Exception:

            logger.exception(
                "Error generating embedding"
            )

            raise

    # ==========================================================
    # Batch Embedding
    # ==========================================================

    def generate_embeddings_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """

        if not texts:
            return []

        try:

            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False,
            )

            results = embeddings.tolist()

            for index, embedding in enumerate(results):

                if len(embedding) != self.vector_size:

                    raise ValueError(
                        f"Invalid embedding dimension at index "
                        f"{index}: "
                        f"expected={self.vector_size}, "
                        f"actual={len(embedding)}"
                    )

            return results

        except Exception:

            logger.exception(
                "Error generating batch embeddings"
            )

            raise

    # ==========================================================
    # Point ID
    # ==========================================================

    @staticmethod
    def _point_id(
        chunk: Dict[str, Any],
    ) -> str:
        """
        Generate deterministic UUID for a chunk.
        """

        chunk_id = chunk.get("chunk_id")

        if chunk_id:

            raw = str(chunk_id)

        else:

            metadata = chunk.get("metadata") or {}

            raw = (
                f"{chunk.get('doc_id')}|"
                f"{metadata.get('chunk_index')}|"
                f"{chunk.get('text', '')}"
            )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).digest()

        return str(
            uuid.UUID(
                bytes=digest[:16]
            )
        )

    # ==========================================================
    # Store chunks
    # ==========================================================

    def store_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
    ) -> bool:
        """
        Generate embeddings and store chunks in Qdrant.

        Important:
        Document metadata is also stored in the payload so
        the RAG agent can identify which article a chunk belongs to.
        """

        if not chunks:
            return True

        try:

            # --------------------------------------------------
            # Valid chunks
            # --------------------------------------------------

            valid_chunks = [
                chunk
                for chunk in chunks
                if isinstance(
                    chunk.get("text"),
                    str,
                )
                and chunk["text"].strip()
            ]

            if not valid_chunks:

                logger.warning(
                    "No valid chunks to store"
                )

                return True

            # --------------------------------------------------
            # Generate embeddings
            # --------------------------------------------------

            texts = [
                chunk["text"]
                for chunk in valid_chunks
            ]

            embeddings = self.generate_embeddings_batch(
                texts
            )

            if len(embeddings) != len(valid_chunks):

                logger.error(
                    "Embedding count mismatch: "
                    "chunks=%d embeddings=%d",
                    len(valid_chunks),
                    len(embeddings),
                )

                return False

            # --------------------------------------------------
            # Build Qdrant points
            # --------------------------------------------------

            points: List[PointStruct] = []

            for chunk, embedding in zip(
                valid_chunks,
                embeddings,
            ):

                chunk["embedding"] = embedding

                metadata = (
                    chunk.get("metadata")
                    or {}
                )

                # ----------------------------------------------
                # Document metadata
                # ----------------------------------------------

                document_title = (
                    chunk.get("document_title")
                    or metadata.get("document_title")
                    or metadata.get("title")
                    or ""
                )

                document_filename = (
                    chunk.get("document_filename")
                    or metadata.get("document_filename")
                    or metadata.get("filename")
                    or ""
                )

                document_path = (
                    chunk.get("document_path")
                    or metadata.get("document_path")
                    or ""
                )

                # ----------------------------------------------
                # Payload
                # ----------------------------------------------

                payload = {
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),

                    "doc_id": chunk.get(
                        "doc_id"
                    ),

                    "text": chunk.get(
                        "text",
                        "",
                    ),

                    "document_title": document_title,

                    "document_filename": document_filename,

                    "document_path": document_path,

                    "chunk_index": metadata.get(
                        "chunk_index"
                    ),

                    "total_chunks": metadata.get(
                        "total_chunks"
                    ),

                    "char_count": metadata.get(
                        "char_count"
                    ),

                    "token_count": metadata.get(
                        "token_count"
                    ),
                }

                point = PointStruct(
                    id=self._point_id(chunk),
                    vector=embedding,
                    payload=payload,
                )

                points.append(point)

            if not points:
                return False

            # --------------------------------------------------
            # Upsert
            # --------------------------------------------------

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

            document_ids = sorted(
                {
                    chunk.get("doc_id")
                    for chunk in valid_chunks
                    if chunk.get("doc_id") is not None
                }
            )

            logger.info(
                "Stored %d chunks in Qdrant. Documents=%s",
                len(points),
                document_ids,
            )

            return True

        except Exception:

            logger.exception(
                "Error storing chunks in Qdrant"
            )

            return False

    # ==========================================================
    # Search
    # ==========================================================

    def search(
        self,
        query: str,
        limit: int = 5,
        doc_id: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic vector search in Qdrant.
        """

        if not query or not query.strip():
            return []

        try:

            query_embedding = self.generate_embedding(
                query
            )

            query_filter = self._build_document_filter(
                doc_id=doc_id,
                document_ids=document_ids,
            )

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            results: List[Dict[str, Any]] = []

            for point in response.points:

                payload = point.payload or {}

                results.append(
                    {
                        "score": float(
                            point.score
                            if point.score is not None
                            else 0.0
                        ),

                        "text": payload.get(
                            "text",
                            "",
                        ),

                        "chunk_id": payload.get(
                            "chunk_id"
                        ),

                        "doc_id": payload.get(
                            "doc_id"
                        ),

                        "document_title": payload.get(
                            "document_title",
                            "",
                        ),

                        "document_filename": payload.get(
                            "document_filename",
                            "",
                        ),

                        "chunk_index": payload.get(
                            "chunk_index"
                        ),

                        "total_chunks": payload.get(
                            "total_chunks"
                        ),

                        "type": "vector",
                    }
                )

            logger.info(
                "Vector search completed: query=%r "
                "results=%d doc_id=%s document_ids=%s",
                query,
                len(results),
                doc_id,
                document_ids,
            )

            return results

        except Exception:

            logger.exception(
                "Error searching Qdrant"
            )

            return []

    # ==========================================================
    # Search per document
    # ==========================================================

    def search_each_document(
        self,
        query: str,
        document_ids: List[int],
        limit_per_document: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search each document independently.

        This is important for comparison questions.
        """

        if not query or not query.strip():
            return []

        if not document_ids:
            return []

        all_results: List[Dict[str, Any]] = []

        for document_id in document_ids:

            results = self.search(
                query=query,
                limit=limit_per_document,
                doc_id=document_id,
            )

            for result in results:
                result["retrieval_document_id"] = document_id

            all_results.extend(results)

        # Highest score first
        all_results.sort(
            key=lambda item: item.get(
                "score",
                0.0,
            ),
            reverse=True,
        )

        return all_results

    # ==========================================================
    # Filter
    # ==========================================================

    @staticmethod
    def _build_document_filter(
        doc_id: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
    ) -> Optional[Filter]:
        """
        Build Qdrant filter for document IDs.
        """

        if doc_id is not None:

            return Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(
                            value=doc_id
                        ),
                    )
                ]
            )

        if document_ids:

            return Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(
                            any=document_ids
                        ),
                    )
                ]
            )

        return None

    # ==========================================================
    # Delete document
    # ==========================================================

    def delete_document(
        self,
        doc_id: int,
    ) -> bool:
        """
        Delete all chunks belonging to a document.
        """

        try:

            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(
                            value=doc_id
                        ),
                    )
                ]
            )

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=query_filter,
                wait=True,
            )

            logger.info(
                "Deleted document %s from Qdrant",
                doc_id,
            )

            return True

        except Exception:

            logger.exception(
                "Error deleting document %s from Qdrant",
                doc_id,
            )

            return False

    # ==========================================================
    # Collection Info
    # ==========================================================

    def get_collection_info(
        self,
    ) -> Dict[str, Any]:

        try:

            info = self.client.get_collection(
                self.collection_name
            )

            vectors_config = (
                info.config.params.vectors
            )

            vector_size = getattr(
                vectors_config,
                "size",
                self.vector_size,
            )

            return {
                "name": self.collection_name,
                "points_count": (
                    info.points_count or 0
                ),
                "vector_size": vector_size,
            }

        except Exception:

            logger.exception(
                "Error getting collection info"
            )

            return {}

    # ==========================================================
    # Health check
    # ==========================================================

    def health_check(self) -> bool:

        try:

            self.client.get_collections()

            return True

        except Exception:

            logger.exception(
                "Qdrant health check failed"
            )

            return False