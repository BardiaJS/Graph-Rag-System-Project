"""
Background ingestion job (runs inside the RQ worker, not the API process).

Heavy imports (docling, torch, sentence-transformers) are done lazily inside the job,
so the API can import this module (for the callbacks) without loading any models.

Two kinds of job:

  process_document_job   converting -> chunking -> figures -> embedding -> graph -> done
                         The document is searchable the moment this finishes (about a
                         minute), and the callback to Laravel fires then.

  extract_entities_job   Entity/relation extraction, queued by the job above on its OWN
                         queue. One LLM call per chunk, so it takes tens of minutes; it
                         enriches the graph later and never blocks ingestion or search.
"""
import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from rq import get_current_job

from app.task_queue import (
    ENTITY_JOB_TIMEOUT,
    FAILURE_TTL,
    RESULT_TTL,
    entities_queue,
    entity_cancel_key,
    entity_job_key,
    redis_conn,
)

logger = logging.getLogger(__name__)

CALLBACK_URL = os.getenv("LARAVEL_CALLBACK_URL")  # fixed in config, NOT taken from requests (SSRF)
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
DEBUG_PRINTS = os.getenv("DEBUG_PRINTS", "1") == "1"
ENABLE_ENTITY_EXTRACTION = os.getenv("ENABLE_ENTITY_EXTRACTION", "1") == "1"


# ----------------------------------------------------------------- helpers

def _stage(stage: str, **extra: Any) -> None:
    """Publish progress so GET /jobs/{id} can show it."""
    logger.info(f"[stage] {stage} {extra or ''}")
    job = get_current_job()
    if job is not None:
        job.meta.update({"stage": stage, **extra})
        job.save_meta()


def _debug(fn, *args, **kwargs) -> None:
    if not DEBUG_PRINTS:
        return
    try:
        fn(*args, **kwargs)
    except Exception as e:
        logger.warning(f"debug helper {fn.__name__} failed: {e}")


# --------------------------------------------------------------------- job

def process_document_job(
    document_id: int,
    file_path: str,
    user_id: str,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert -> chunk -> Qdrant -> Neo4j graph (+ figures), then queue entity extraction.
    Safe to re-run: every store is purged/rebuilt for this document first. Returns a SMALL summary
    (it is stored in Redis); the markdown is written to disk instead."""
    from app.document_processor import DocumentProcessor
    from app.services.debug_service import (
        print_chunks_summary,
        print_neo4j_graph,
        print_qdrant_points,
    )
    from app.services.embedding_service import upload_chunks
    from app.services.document_service import markdown_path as markdown_file
    from app.services.document_service import purge_qdrant
    from app.services.figure_service import (
        figure_dir,
        purge_figure_files,
        store_figures,
    )
    from app.services.graph_service import build_graph, delete_document_graph

    document_id = int(document_id)
    user_id = str(user_id)

    # An entity job from a previous run would otherwise attach stale entities to the chunks
    # this run is about to recreate (chunk keys are identical). Tell it to stop.
    _cancel_entity_work(document_id)

    processor = DocumentProcessor(file_path)
    ocr = processor.ocr_plan()  # cheap text-layer check, done before the slow conversion
    _stage("converting", ocr=ocr["use_ocr"], pdf_kind=ocr["kind"])
    markdown = processor.docling_service()

    markdown_path = markdown_file(document_id)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    _stage("chunking")
    chunks = processor.chunking_service()

    summary: Dict[str, Any] = {
        "document_id": document_id,
        "chunks": len(chunks),
        "markdown_path": str(markdown_path),
        "ocr": ocr,
        "entities": None,
        "entity_error": None,
        "figures": None,
        "figure_error": None,
    }

    # Figures: extract the images now (the converted document is cached, so this costs
    # no extra parsing). Linking to chunks happens after the graph exists. A failure here
    # must not stop text indexing, so it is recorded instead of raised.
    _stage("figures")
    figures = []
    purge_figure_files(document_id)  # a re-run never leaves stale images
    try:
        figures = processor.figures_service(figure_dir(document_id))
    except Exception as e:
        logger.error(f"Figure extraction failed for document {document_id}: {e}")
        summary["figure_error"] = str(e)

    if not chunks:
        # Nothing extractable (e.g. scanned PDF with OCR disabled): clear old data.
        purge_qdrant(document_id)
        delete_document_graph(document_id)
        purge_figure_files(document_id)
        _stage("done")
        summary["warning"] = (
            "No text chunks extracted"
            + (" even with OCR" if ocr["use_ocr"] else
               " (PDF looks image-only: set DOCLING_OCR=on or install an OCR engine)"
               if ocr["kind"] != "digital" else "")
        )
        return summary

    _debug(print_chunks_summary, chunks)

    _stage("embedding", total=len(chunks))
    purge_qdrant(document_id)
    summary["qdrant_points"] = upload_chunks(chunks, user_id=user_id, document_id=document_id)
    _debug(print_qdrant_points, document_id=document_id, limit=5)

    _stage("graph")
    build_graph(
        document_id=document_id,
        chunks=chunks,
        title=f"Document {document_id}",
        user_id=user_id,
    )

    if figures:
        try:
            summary["figures"] = store_figures(document_id, figures, chunks)
        except Exception as e:
            logger.error(f"Storing figures failed for document {document_id}: {e}")
            summary["figure_error"] = str(e)
    else:
        summary["figures"] = {"figures": 0, "links": 0, "unlinked": 0}

    if ENABLE_ENTITY_EXTRACTION:
        # Not done here any more: it takes tens of minutes. Queue it and finish now, so the
        # document becomes searchable (and Laravel is told) as soon as the graph exists.
        try:
            summary["entities"] = enqueue_entities(document_id)
        except Exception as e:
            logger.error(f"Could not queue entity extraction for document {document_id}: {e}")
            summary["entity_error"] = f"Could not queue entity extraction: {e}"

    _debug(print_neo4j_graph, document_id=document_id, limit=10)
    _stage("done")
    return summary


# ---------------------------------------------------------------- entities

def _cancel_entity_work(document_id: int) -> None:
    try:
        redis_conn.set(entity_cancel_key(document_id), "1", ex=3600)
    except Exception as e:
        logger.warning(f"Could not set entity cancel flag for document {document_id}: {e}")


def enqueue_entities(document_id: int) -> Dict[str, Any]:
    """Queue entity extraction for a freshly ingested document."""
    redis_conn.delete(entity_cancel_key(document_id))  # lift the stop flag set at job start

    # A unique id per run: re-using one id while an older job is still running would
    # overwrite that job's record.
    job_id = f"ent-{document_id}-{int(time.time())}"
    entities_queue.enqueue(
        extract_entities_job,
        document_id=document_id,
        job_id=job_id,
        job_timeout=ENTITY_JOB_TIMEOUT,
        result_ttl=RESULT_TTL,
        failure_ttl=FAILURE_TTL,
        meta={"document_id": document_id, "stage": "queued"},
    )
    redis_conn.set(entity_job_key(document_id), job_id, ex=7 * 86400)
    return {"status": "queued", "job_id": job_id}


def extract_entities_job(document_id: int) -> Dict[str, Any]:
    """Extract entities/relations for an already-ingested document.

    Reads the chunks back from Neo4j (nothing large travels through Redis), stops at once
    if the document is deleted or re-processed meanwhile, and raises if the language
    model is unavailable so the failure shows up in GET /documents/{id}."""
    from app.services.entity_service import extract_and_store_entities
    from app.services.graph_service import get_chunk_texts

    document_id = int(document_id)
    _stage("entities", progress=0, total=0)

    chunks = get_chunk_texts(document_id)
    if not chunks:
        return {"document_id": document_id, "warning": "document is not in the graph (deleted?)"}

    _stage("entities", progress=0, total=len(chunks))
    stats = extract_and_store_entities(
        document_id,
        chunks,
        on_progress=lambda done, total: _stage("entities", progress=done, total=total),
        should_stop=lambda: bool(redis_conn.exists(entity_cancel_key(document_id))),
    )
    _stage("done")
    return {"document_id": document_id, **stats}


# --------------------------------------------------------------- callbacks
# RQ invokes these in the worker when a job finishes, fails, or hits its timeout.

def _notify(payload: Dict[str, Any]) -> None:
    if not CALLBACK_URL:
        return
    headers = {"X-API-Key": INTERNAL_API_KEY} if INTERNAL_API_KEY else {}
    try:
        resp = requests.post(CALLBACK_URL, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:  # a down Laravel must never turn a good job into a failure
        logger.warning(f"Callback to {CALLBACK_URL} failed: {e}")


def _callback_fields(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Flat integers/strings, matching the Laravel webhook's validation rules."""
    r = result or {}
    entities = r.get("entities") or {}
    figures = r.get("figures") or {}
    ocr = r.get("ocr") or {}
    return {
        "pages": ocr.get("pages_total"),
        "chunks": int(r.get("chunks") or 0),
        "entities": int(entities.get("entities") or 0),
        "figures": int(figures.get("figures") or 0),
        "qdrant_points": int(r.get("qdrant_points") or 0),
        "ocr_used": bool(ocr.get("use_ocr")),
        "pdf_kind": ocr.get("kind"),
        "entity_error": r.get("entity_error"),
        "figure_error": r.get("figure_error"),
        "warning": r.get("warning"),
    }


def on_success(job, connection, result, *args, **kwargs):
    _notify({
        "job_id": job.id,
        "document_id": job.meta.get("document_id"),
        "status": "success",
        **_callback_fields(result),
    })


def on_failure(job, connection, exc_type, value, traceback):
    _notify({
        "job_id": job.id,
        "document_id": job.meta.get("document_id"),
        "status": "failed",
        "error": str(value) or getattr(exc_type, "__name__", "error"),
    })
