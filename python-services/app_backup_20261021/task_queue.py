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