"""
FastAPI Service for Graph-RAG System

Handles:
- Document processing
- Agentic RAG question answering
- Qdrant / Neo4j / Ollama health checks
- Redis document status
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

import redis
import json
import os
import logging
from app.services.document_processor import DocumentProcessor
from dotenv import load_dotenv
# from app.services.chunk_evaluator import ChunkEvaluator
from fastapi.responses import JSONResponse


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production: specify allowed domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Redis Connection
# ============================================================

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)

try:
    redis_client.ping()
    logger.info("✅ Connected to Redis successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")


# ============================================================
# Initialize Agentic RAG Service
# ============================================================

RAG_AVAILABLE = False
rag_service = None

# try:
#     from app.services.agentic_rag_service import (
#         AgenticRAGService as RAGService
#     )

#     rag_service = RAGService()

#     # Debug Qdrant documents on startup
#     try:
#         rag_service.debug_documents()
#     except Exception as e:
#         logger.warning(
#             f"⚠️ Qdrant debug failed during startup: {e}"
#         )

#     RAG_AVAILABLE = True

#     logger.info(
#         "✅ RAG Service initialized successfully"
#     )

# except Exception as e:
#     logger.exception(
#         "❌ Failed to initialize RAG Service"
#     )


# ============================================================
# Pydantic Models
# ============================================================
class ProcessDocumentRequest(BaseModel):
    user_id: int
    session_id: int
    file_path: str
    document_ids: List[int]


@app.post("/process")
async def process_document(request: ProcessDocumentRequest):
    processor = DocumentProcessor(
        request.file_path,
        request.user_id,
        request.document_ids[0],
    )

    doc = processor.extract_document()
    markdown = doc.export_to_markdown()
    doc_dict = doc.export_to_dict()

    processor.extract_image()

    # ==== DEBUG ====
    print("=" * 60)
    print("DOC_DICT TOP-LEVEL KEYS:", list(doc_dict.keys()))
    print("NUM TEXTS:", len(doc_dict.get("texts", [])))
    print("NUM TABLES:", len(doc_dict.get("tables", [])))
    print("NUM PICTURES:", len(doc_dict.get("pictures", [])))
    print("-" * 60)
    print("MARKDOWN PREVIEW (first 3000 chars):")
    print(markdown[:3000])
    print("=" * 60)
    # ==== END DEBUG ====

    return {
        "status": "ok",
        "markdown_preview": markdown[:3000],
        "num_texts": len(doc_dict.get("texts", [])),
        "num_tables": len(doc_dict.get("tables", [])),
        "num_pictures": len(doc_dict.get("pictures", [])),
    }