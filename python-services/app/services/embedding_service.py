import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ← اسم جدید چون مدل عوض شده
COLLECTION_NAME = "graph_rag_chunks_multilingual"

# ← مدل چندزبانه
EMBED_MODEL = "intfloat/multilingual-e5-small"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _model


def ensure_collection():
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(f"Created collection: {COLLECTION_NAME}")


def upload_chunks(chunks: list, user_id: str = None, document_id: int = None):
    if not chunks:
        return 0

    ensure_collection()
    model = get_model()

    # ← e5 نیاز به prefix "passage:" داره
    texts = ["passage: " + c["enriched_text"] for c in chunks]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    import hashlib
    points = []
    for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
        unique_id = int(
            hashlib.md5(f"{document_id}_{idx}".encode()).hexdigest()[:12], 16
        )
        points.append(models.PointStruct(
            id=unique_id,
            vector=vector,
            payload={
                "text": chunk["text"],
                "enriched_text": chunk["enriched_text"],
                "headings": chunk.get("headings", []),
                "page": chunk.get("page"),
                "user_id": user_id,
                "document_id": document_id,
            }
        ))

    client.upload_points(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Uploaded {len(points)} chunks to Qdrant")
    return len(points)


def search(query: str, limit: int = 5, user_id: str = None):
    model = get_model()
    # ← e5 نیاز به prefix "query:" داره
    query_vector = model.encode("query: " + query).tolist()

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
    ).points

    return [
        {
            "text": h.payload["text"],
            "enriched_text": h.payload.get("enriched_text"),
            "headings": h.payload.get("headings", []),
            "page": h.payload.get("page"),
            "score": h.score,
        }
        for h in hits
    ]