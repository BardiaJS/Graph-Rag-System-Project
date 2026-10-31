"""
RQ worker. Run from the project root:

    python -m app.worker

Why SimpleWorker: the default RQ Worker forks a new process for every job. With
torch/docling that means re-initialising heavy state per job (and fork + torch is
fragile). SimpleWorker runs jobs in this one process, so the embedding model is loaded
once. Run ONE worker process: that is also your OOM guard (one document at a time),
replacing the old ThreadPoolExecutor(max_workers=1).
"""
import os

from dotenv import load_dotenv

load_dotenv()  # before any service import (they read env at import time)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import logging

from rq import SimpleWorker

from app.task_queue import QUEUE_NAME, redis_conn

logger = logging.getLogger("worker")


def preload() -> None:
    """Load models once, up front, so the first job isn't slow and a missing
    offline model fails at startup instead of mid-job."""
    from app.services.docling_service import DoclingService
    from app.services.embedding_service import get_model

    get_model()
    DoclingService._get_tokenizer()
    logger.info("Models preloaded")

    # Catch a missing/mistyped Ollama model now, not 150 failed calls into the first upload.
    if os.getenv("ENABLE_ENTITY_EXTRACTION", "1") == "1":
        from app.services import llm_service

        problem = llm_service.check_model()
        if problem:
            logger.warning(f"ENTITY EXTRACTION WILL FAIL: {problem}")
        else:
            logger.info(f"Ollama model '{llm_service.OLLAMA_MODEL}' is available")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    preload()
    logger.info(f"Worker listening on queue '{QUEUE_NAME}'")
    SimpleWorker([QUEUE_NAME], connection=redis_conn).work(with_scheduler=False)
