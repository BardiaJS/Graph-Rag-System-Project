import os

# 1) Load .env BEFORE importing any app module: services read os.getenv(...) at import
#    time, so loading .env later would silently ignore QDRANT_URL / NEO4J_* / OLLAMA_* / REDIS_*.
from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import hmac
import logging
from pathlib import Path
from typing import List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from rq.exceptions import NoSuchJobError
from rq.job import Job

from app.jobs import on_failure, on_success, process_document_job
from app.services.debug_service import print_search_results
from app.services.embedding_service import search, search_with_graph
from app.services.figure_service import figure_path
from app.services.retrieval_service import retrieve
from app.services.verify_service import verify_answer
from app.services.document_service import count_qdrant_points, delete_document, qdrant_owner
from app.services.graph_service import get_document, get_figures, list_documents
from app.task_queue import FAILURE_TTL, JOB_TIMEOUT, RESULT_TTL, queue, redis_conn

# ============================================================
# Config / logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LARAVEL_STORAGE_PATH = os.getenv(
    "LARAVEL_STORAGE_PATH",
    "/home/bardia/Desktop/graph-rag/backend/storage/app/public",  # dev default only
)
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")  # unset = open (dev only)
# The LLM agent loop belongs in Laravel; this server-side copy is opt-in.
ENABLE_AGENT_ENDPOINT = os.getenv("ENABLE_AGENT_ENDPOINT", "0") == "1"
DEBUG_PRINTS = os.getenv("DEBUG_PRINTS", "1") == "1"

ACTIVE_STATUSES = {"queued", "started", "deferred", "scheduled"}


def debug(fn, *args, **kwargs):
    """Debug printers must never be able to break a request."""
    if not DEBUG_PRINTS:
        return
    try:
        fn(*args, **kwargs)
    except Exception as e:
        logger.warning(f"debug helper {fn.__name__} failed: {e}")


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="Graph RAG - PDF Processor")

_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],  # credentials + wildcard origin is invalid/unsafe
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Shared secret between Laravel and this service (header: X-API-Key)."""
    if INTERNAL_API_KEY and not hmac.compare_digest(x_api_key or "", INTERNAL_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.on_event("startup")
def check_redis():
    try:
        redis_conn.ping()
        logger.info("Connected to Redis successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")


# ============================================================
# Helpers
# ============================================================

def resolve_upload_path(relative_path: str) -> str:
    """Resolve a Laravel-relative path; refuse anything outside the storage dir."""
    base = Path(LARAVEL_STORAGE_PATH).resolve()
    full = (base / relative_path).resolve()
    if base not in full.parents:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return str(full)


def job_id_for(document_id: int) -> str:
    return f"doc-{document_id}"


def _status_str(job: Job) -> str:
    s = job.get_status()
    return str(getattr(s, "value", s))


def _fetch_job(job_id: str) -> Optional[Job]:
    try:
        return Job.fetch(job_id, connection=redis_conn)
    except NoSuchJobError:
        return None


# ============================================================
# POST /process  -> enqueue, return immediately
# ============================================================

class ProcessRequest(BaseModel):
    user_id: Union[str, int]
    session_id: Union[str, int]
    file_path: str
    document_ids: List[Union[int, str]]


@app.post("/process", status_code=202, dependencies=[Depends(require_api_key)])
def process_document(request: ProcessRequest):
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="document_ids required")
    try:
        doc_id = int(request.document_ids[0])
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="document_ids must be integers")
    if len(request.document_ids) > 1:
        logger.warning(
            f"{len(request.document_ids)} document_ids sent for one file; "
            f"only {doc_id} is indexed"
        )

    full_path = resolve_upload_path(request.file_path)
    job_id = job_id_for(doc_id)

    # Idempotent enqueue: a retry or double-click while the document is already queued
    # or running returns the existing job instead of processing it twice.
    existing = _fetch_job(job_id)
    if existing is not None and _status_str(existing) in ACTIVE_STATUSES:
        return {
            "status": "queued",
            "job_id": job_id,
            "document_id": doc_id,
            "deduplicated": True,
        }

    try:
        queue.enqueue(
            process_document_job,
            document_id=doc_id,
            file_path=full_path,
            user_id=str(request.user_id),
            session_id=str(request.session_id),
            job_id=job_id,
            job_timeout=JOB_TIMEOUT,
            result_ttl=RESULT_TTL,
            failure_ttl=FAILURE_TTL,
            on_success=on_success,
            on_failure=on_failure,
            meta={"document_id": doc_id, "stage": "queued"},
        )
    except Exception as e:
        logger.exception("Could not enqueue job (is Redis up?)")
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {e}")

    logger.info(f"Enqueued {job_id} for {full_path}")
    return {"status": "queued", "job_id": job_id, "document_id": doc_id}


# ============================================================
# GET /jobs/{job_id}  -> poll status
# ============================================================

@app.get("/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def job_status(job_id: str):
    job = _fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found (or expired)")

    status = _status_str(job)
    body = {
        "job_id": job.id,
        "document_id": job.meta.get("document_id"),
        "status": status,  # queued | started | finished | failed | ...
        "stage": job.meta.get("stage"),
        "progress": job.meta.get("progress"),
        "total": job.meta.get("total"),
    }
    if status == "queued":
        body["position"] = job.get_position()
    if status == "finished":
        body["result"] = job.result
    if status == "failed":
        lines = (job.exc_info or "").strip().splitlines()
        body["error"] = lines[-1] if lines else "Job failed"
    return body


# ============================================================
# Documents: list / detail / delete  (FR-10)
# ============================================================

def _check_owner(document_id: int, user_id: Optional[str]) -> Optional[dict]:
    """Return the graph record (or None). If user_id is given and the document belongs
    to someone else, answer 404 (not 403) so existence isn't revealed.

    Ownership comes from the graph; if the graph half of an interrupted ingestion is
    missing, it falls back to the user_id stored on the Qdrant points."""
    doc = get_document(document_id)
    if user_id is None:
        return doc
    owner = doc.get("user_id") if doc else qdrant_owner(document_id)
    if doc is None and owner is None:
        return None  # nothing exists, so nothing to protect
    if owner is None or str(owner) != str(user_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@app.get("/documents", dependencies=[Depends(require_api_key)])
def documents_list(user_id: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Documents the service holds, with sizes. Laravel can diff this against its own
    table to find orphans. Always pass user_id for user-facing calls."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    return {
        "status": "success",
        "documents": list_documents(user_id, limit, offset),
        "limit": limit,
        "offset": offset,
    }


@app.get("/documents/{document_id}", dependencies=[Depends(require_api_key)])
def document_detail(document_id: int, user_id: Optional[str] = None):
    doc = _check_owner(document_id, user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    points = count_qdrant_points(document_id)
    doc["qdrant_points"] = points
    # One vector per chunk. A mismatch means the two stores disagree (e.g. a job that
    # failed half-way): re-run /process for that document to repair it.
    doc["consistent"] = None if points is None else points == doc["chunks"]

    job = _fetch_job(job_id_for(document_id))
    doc["job"] = (
        {"status": _status_str(job), "stage": job.meta.get("stage")} if job else None
    )
    return {"status": "success", "document": doc}


@app.delete("/documents/{document_id}", dependencies=[Depends(require_api_key)])
def document_delete(document_id: int, user_id: Optional[str] = None):
    """Remove a document from Qdrant, Neo4j and disk. Idempotent: deleting something
    that is already gone succeeds."""
    _check_owner(document_id, user_id)

    # Never delete underneath a running ingestion: it would recreate the data a moment
    # later. A job that is still waiting in the queue is simply cancelled.
    job = _fetch_job(job_id_for(document_id))
    if job is not None:
        if _status_str(job) == "started":
            raise HTTPException(
                status_code=409,
                detail="Document is being processed; try again when the job finishes",
            )
        try:
            job.delete(remove_from_queue=True)
        except Exception as e:
            logger.warning(f"Could not remove job record for document {document_id}: {e}")

    result = delete_document(document_id)
    if not result["complete"]:
        # Surface partial failure loudly (a 2xx would be read as success). The call is
        # idempotent, so the caller can just retry it.
        raise HTTPException(
            status_code=500,
            detail={"status": "partial", **result},
        )
    return {"status": "success", **result}


# ============================================================
# Figures
# ============================================================

def figure_url(document_id: int, figure_index: int) -> str:
    return f"/documents/{document_id}/figures/{figure_index}/image"


@app.get("/documents/{document_id}/figures", dependencies=[Depends(require_api_key)])
def list_figures(document_id: int):
    figures = get_figures(document_id)
    for f in figures:
        f["image_url"] = figure_url(document_id, f["figure_index"])
    return {"status": "success", "document_id": document_id, "figures": figures}


@app.get(
    "/documents/{document_id}/figures/{figure_index}/image",
    dependencies=[Depends(require_api_key)],
)
def figure_image(document_id: int, figure_index: int):
    # Path is built from the two integers only (see figure_service.figure_path).
    path = figure_path(document_id, figure_index)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(path, media_type="image/png")


# ============================================================
# POST /search  (raw retrieval, no LLM)
# ============================================================

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    document_id: Optional[int] = None
    use_graph: bool = True


@app.post("/search", dependencies=[Depends(require_api_key)])
def search_endpoint(request: SearchRequest):  # plain def -> FastAPI threadpool
    if request.document_id is None:
        logger.warning("/search called without document_id: searching ALL documents")

    if request.use_graph:
        results = search_with_graph(
            query=request.query,
            limit=request.limit,
            document_id=request.document_id,
        )
    else:
        results = search(
            query=request.query,
            limit=request.limit,
            document_id=request.document_id,
        )

    debug(print_search_results, request.query, results)
    return {"status": "success", "results": results}


# ============================================================
# POST /retrieve  (stateless evidence retrieval for the Laravel agent loop)
# ============================================================

class RetrieveQuery(BaseModel):
    q: str
    document_ids: Optional[List[int]] = None  # narrowed to the allowed set below


class RetrieveRequest(BaseModel):
    queries: List[RetrieveQuery]
    document_ids: List[int]  # the caller's allowed documents: the hard scope limit


@app.post("/retrieve", dependencies=[Depends(require_api_key)])
def retrieve_endpoint(request: RetrieveRequest):
    """Vector + neighbour + shared-entity + figure retrieval. No LLM involved.
    Laravel runs plan -> retrieve -> grade -> retrieve -> synthesize itself, merges the
    returned evidence by `key`, assigns its own source ids (S1, S2, ...) and finally
    calls /verify."""
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="document_ids required")
    queries = [
        {"q": q.q, "document_ids": q.document_ids}
        for q in request.queries
        if q.q.strip()
    ]
    if not queries:
        raise HTTPException(status_code=400, detail="at least one non-empty query required")
    try:
        return {"status": "success", **retrieve(queries, request.document_ids)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Retrieve failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# POST /verify  (deterministic citation + figure-reference check)
# ============================================================

class VerifySource(BaseModel):
    sid: str                      # "S1", "S2", ... (S + number)
    text: str = ""
    figure_numbers: List[str] = []  # numbers of figures linked to this source


class VerifyRequest(BaseModel):
    answer: str
    sources: List[VerifySource]


@app.post("/verify", dependencies=[Depends(require_api_key)])
def verify_endpoint(request: VerifyRequest):
    sources = [
        {"sid": s.sid, "text": s.text, "figure_numbers": list(s.figure_numbers)}
        for s in request.sources
    ]
    return {"status": "success", **verify_answer(request.answer, sources)}


# ============================================================
# POST /ask  (OPTIONAL server-side agent; off unless ENABLE_AGENT_ENDPOINT=1)
# ============================================================

if ENABLE_AGENT_ENDPOINT:
    from dataclasses import asdict

    import requests

    from app.services.agent_service import answer_question

    class AskRequest(BaseModel):
        question: str
        document_ids: List[int]  # required on purpose: no document_ids, no search

    @app.post("/ask", dependencies=[Depends(require_api_key)])
    def ask_endpoint(request: AskRequest):
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="question required")
        if not request.document_ids:
            raise HTTPException(status_code=400, detail="document_ids required")
        try:
            result = answer_question(request.question, request.document_ids)
            return {"status": "success", **asdict(result)}
        except requests.RequestException as e:
            logger.error(f"LLM unreachable: {e}")
            raise HTTPException(status_code=503, detail="Language model unavailable")
        except Exception as e:
            logger.exception("Ask failed")
            raise HTTPException(status_code=500, detail=str(e))
