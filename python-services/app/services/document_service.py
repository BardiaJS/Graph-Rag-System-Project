"""
Document lifecycle across every store that holds a document's data:

    Qdrant   chunk vectors                       (purge_qdrant)
    Neo4j    Document/Section/Chunk/Figure,      (graph_service.delete_document_graph;
             Entity mentions and relations        also removes entities nobody mentions)
    Disk     figure PNGs, markdown export        (figure_service / markdown_path)

Everything is imported lazily so the API process doesn't pull in qdrant/torch just to
import this module, and so a store that is down only affects its own step.
"""
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))


def markdown_path(document_id: int) -> Path:
    return PROCESSED_DIR / f"{int(document_id)}.md"


# ------------------------------------------------------------------ Qdrant

def _doc_filter(document_id: int):
    from qdrant_client import models

    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=int(document_id)),
            )
        ]
    )


def purge_qdrant(document_id: int) -> None:
    """Delete every point of a document (used before re-indexing and on delete)."""
    from qdrant_client import models

    from app.services import embedding_service as emb

    emb.ensure_collection()
    emb.client.delete(
        collection_name=emb.COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=_doc_filter(document_id)),
    )


def count_qdrant_points(document_id: int) -> Optional[int]:
    """Points stored for a document, or None if Qdrant can't be queried
    (None means 'unknown', which must not be confused with 'zero')."""
    from app.services import embedding_service as emb

    try:
        return emb.client.count(
            collection_name=emb.COLLECTION_NAME,
            count_filter=_doc_filter(document_id),
            exact=True,
        ).count
    except Exception as e:
        logger.warning(f"Qdrant count failed for document {document_id}: {e}")
        return None


def qdrant_owner(document_id: int) -> Optional[str]:
    """user_id recorded on a document's points. Lets ownership be checked even when the
    graph half of an interrupted ingestion is missing."""
    from app.services import embedding_service as emb

    try:
        points, _ = emb.client.scroll(
            collection_name=emb.COLLECTION_NAME,
            scroll_filter=_doc_filter(document_id),
            limit=1,
            with_payload=["user_id"],
            with_vectors=False,
        )
        return points[0].payload.get("user_id") if points else None
    except Exception as e:
        logger.warning(f"Qdrant owner lookup failed for document {document_id}: {e}")
        return None


# ------------------------------------------------------------------ delete

def delete_document(document_id: int) -> Dict[str, Any]:
    """Remove a document from every store. Idempotent, and one failing store never
    prevents the others from being cleaned: errors are collected, not raised, so the
    caller can report 'partial' and simply call delete again."""
    from app.services.figure_service import purge_figure_files
    from app.services.graph_service import delete_document_graph

    document_id = int(document_id)
    points_before = count_qdrant_points(document_id)

    def remove_markdown():
        markdown_path(document_id).unlink(missing_ok=True)

    # Qdrant first: the moment it succeeds, search can no longer return this document.
    steps = [
        ("qdrant", lambda: purge_qdrant(document_id)),
        ("graph", lambda: delete_document_graph(document_id)),
        ("figures", lambda: purge_figure_files(document_id)),
        ("markdown", remove_markdown),
    ]

    removed: Dict[str, bool] = {}
    errors = []
    for name, fn in steps:
        try:
            fn()
            removed[name] = True
        except Exception as e:
            logger.error(f"Delete of document {document_id}: {name} step failed: {e}")
            removed[name] = False
            errors.append({"store": name, "error": str(e)})

    return {
        "document_id": document_id,
        "qdrant_points_deleted": points_before if removed.get("qdrant") else None,
        "removed": removed,
        "errors": errors,
        "complete": not errors,
    }
