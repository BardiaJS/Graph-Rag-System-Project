"""
Shared RQ connection/queue. (Named task_queue to avoid clashing with stdlib `queue`.)

Two Redis gotchas this module avoids:
  * RQ stores binary/pickled data, so the connection must NOT use decode_responses=True
    (the old client in main.py did; it is fine for pings but breaks RQ).
  * Do NOT set socket_timeout. The worker waits on the queue with a blocking pop that
    is longer than a short socket timeout (the old client used 5s), which would raise
    TimeoutError in the worker loop.
"""
import os

from redis import Redis
from rq import Queue

QUEUE_NAME = os.getenv("RQ_QUEUE", "graph_rag")
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", 3600))       # max seconds per document
RESULT_TTL = int(os.getenv("JOB_RESULT_TTL", 86400))    # keep results/status 24h
FAILURE_TTL = int(os.getenv("JOB_FAILURE_TTL", 86400))

redis_conn = Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD") or None,
    socket_connect_timeout=5,
)

queue = Queue(QUEUE_NAME, connection=redis_conn, default_timeout=JOB_TIMEOUT)

# Entity extraction is slow (one LLM call per chunk: tens of minutes), so it runs as its OWN
# job on its OWN queue. A document is searchable as soon as ingestion finishes; entities
# enrich it later, and a long entity run never blocks other uploads.
ENTITIES_QUEUE_NAME = os.getenv("RQ_ENTITIES_QUEUE", "graph_rag_entities")
ENTITY_JOB_TIMEOUT = int(os.getenv("ENTITY_JOB_TIMEOUT", 6 * 3600))
entities_queue = Queue(ENTITIES_QUEUE_NAME, connection=redis_conn, default_timeout=ENTITY_JOB_TIMEOUT)


def entity_cancel_key(document_id: int) -> str:
    """While this key exists, entity work for the document must stop and write nothing
    (set on delete and on re-processing)."""
    return f"graphrag:entities:cancel:{int(document_id)}"


def entity_job_key(document_id: int) -> str:
    """Holds the id of the most recent entity job for a document."""
    return f"graphrag:entities:job:{int(document_id)}"
