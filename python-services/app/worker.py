"""
RQ worker. Run from the project root:

    python -m app.worker              # ingestion worker: PDFs -> vectors + graph + figures
    python -m app.worker --entities   # entity worker: slow LLM enrichment, runs in parallel

Run exactly ONE of each. The entity worker is optional: without it documents are fully
searchable and entity jobs simply wait in their queue.

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

import argparse
import logging

from rq import SimpleWorker

from app.task_queue import ENTITIES_QUEUE_NAME, QUEUE_NAME, redis_conn

logger = logging.getLogger("worker")


def preload(entities_only: bool = False) -> None:
    """Load models once, up front, so the first job isn't slow and a missing
    offline model fails at startup instead of mid-job."""
    if not entities_only:  # the entity worker never embeds or parses PDFs
        from app.services.docling_service import DoclingService
        from app.services.embedding_service import get_model

        get_model()
        DoclingService._get_tokenizer()
        logger.info("Models preloaded")

    # Catch a missing/mistyped Ollama model now, not 150 failed calls into the first upload.
    if entities_only or os.getenv("ENABLE_ENTITY_EXTRACTION", "1") == "1":
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", action="store_true",
                        help="process the entity-extraction queue instead of the ingestion queue")
    args = parser.parse_args()

    queue_name = ENTITIES_QUEUE_NAME if args.entities else QUEUE_NAME
    preload(entities_only=args.entities)
    logger.info(f"Worker listening on queue '{queue_name}'")
    SimpleWorker([queue_name], connection=redis_conn).work(with_scheduler=False)
