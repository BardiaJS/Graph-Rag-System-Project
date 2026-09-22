from typing import List, Dict, Any, Optional
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
)


class VectorService:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "graph_rag_chunks",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name

        self.client = QdrantClient(
            host=self.host,
            port=self.port,
        )

    def create_collection(
        self,
        dimensions: int,
        recreate: bool = False,
    ):
        exists = self.client.collection_exists(
            self.collection_name
        )

        if exists and recreate:
            self.client.delete_collection(
                self.collection_name
            )
            exists = False

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimensions,
                    distance=Distance.COSINE,
                ),
            )

    def _make_point_id(
        self,
        document_id: Any,
        chunk_id: Any,
        chunk_index: Any,
    ) -> str:

        raw_id = (
            f"document:{document_id}:"
            f"chunk:{chunk_id}:"
            f"index:{chunk_index}"
        )

        return str(
            uuid5(NAMESPACE_URL, raw_id)
        )

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: Optional[Any] = None,
        user_id: Optional[Any] = None,
        session_id: Optional[Any] = None,
    ) -> int:

        if not chunks:
            return 0

        points = []

        for chunk in chunks:
            embedding = chunk.get("embedding")

            if not embedding:
                raise ValueError(
                    "Chunk does not contain an embedding."
                )

            chunk_document_id = chunk.get(
                "document_id",
                document_id,
            )

            chunk_id = chunk.get("chunk_id")
            chunk_index = chunk.get("chunk_index")

            point_id = self._make_point_id(
                document_id=chunk_document_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
            )

            payload = dict(chunk)

            # embedding نباید داخل payload ذخیره شود
            payload.pop("embedding", None)

            if user_id is not None:
                payload["user_id"] = user_id

            if session_id is not None:
                payload["session_id"] = session_id

            if chunk_document_id is not None:
                payload["document_id"] = chunk_document_id

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return len(points)

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        document_ids: Optional[List[Any]] = None,
    ):
        query_filter = None

        if document_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(
                            any=document_ids
                        ),
                    )
                ]
            )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

        return result.points

    def get_info(self) -> Dict[str, Any]:
        info = self.client.get_collection(
            self.collection_name
        )

        return {
            "collection": self.collection_name,
            "points_count": info.points_count,
            "status": str(info.status),
        }