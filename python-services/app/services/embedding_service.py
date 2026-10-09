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

COLLECTION_NAME = "graph_rag_chunks_multilingual"
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
                "chunk_index": idx,   # ← اضافه شد
            }
        ))

    client.upload_points(collection_name=COLLECTION_NAME, points=points)
    logger.info(f"Uploaded {len(points)} chunks to Qdrant")
    return len(points)


def search(query: str, limit: int = 5, user_id: str = None):
    """vector search ساده"""
    model = get_model()
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
            "chunk_index": h.payload.get("chunk_index"),
            "document_id": h.payload.get("document_id"),
            "score": h.score,
        }
        for h in hits
    ]


def search_with_graph(query: str, limit: int = 5, document_id: int = None):
    """
    ۱. vector search
    ۲. برای هر نتیجه، chunkهای همسایه رو از گراف بگیر
    ۳. همه رو برگردون
    """
    from app.services.graph_service import get_chunks_with_context

    # vector search
    results = search(query, limit=limit)

    if not document_id:
        return results

    enriched = []
    seen_indices = set()

    # اول همه chunkهای اصلی
    for r in results:
        idx = r.get("chunk_index")
        if idx is not None:
            enriched.append(r)
            seen_indices.add(idx)

    # بعد همسایه‌ها
    for r in results:
        chunk_idx = r.get("chunk_index")
        if chunk_idx is None:
            continue

        neighbors = get_chunks_with_context(document_id, chunk_idx, context_size=1)

        for n in neighbors:
            if n["idx"] not in seen_indices:
                enriched.append({
                    "text": n["text"],
                    "page": n["page"],
                    "chunk_index": n["idx"],
                    "document_id": document_id,
                    "headings": [],   # ← اضافه کن
                    "score": r["score"] * 0.9,
                    "source": "graph",
                })
                seen_indices.add(n["idx"])

    # مرتب بر اساس score
    enriched.sort(key=lambda x: x["score"], reverse=True)
    return enriched