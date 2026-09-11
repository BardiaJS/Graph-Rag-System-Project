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
from app.services.chunk_evaluator import ChunkEvaluator
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

try:
    from app.services.agentic_rag_service import (
        AgenticRAGService as RAGService
    )

    rag_service = RAGService()

    # Debug Qdrant documents on startup
    try:
        rag_service.debug_documents()
    except Exception as e:
        logger.warning(
            f"⚠️ Qdrant debug failed during startup: {e}"
        )

    RAG_AVAILABLE = True

    logger.info(
        "✅ RAG Service initialized successfully"
    )

except Exception as e:
    logger.exception(
        "❌ Failed to initialize RAG Service"
    )


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

    # ---------- Extraction ----------
    extracted_data = processor.text_extraction()
    extracted_image = processor.image_extraction()
    structure = processor.structure_extraction()

    # ---------- Chunking ----------
    chunk_data = processor.chunk_service(extracted_data["pages"])

    # ---------- Embedding ----------

    embeddings = processor.embedding(chunk_data)

    print("\n========== EMBEDDING RESULTS ==========\n")

    print("TOTAL EMBEDDINGS:", len(embeddings))
    print("VECTOR DIMENSION:", len(embeddings[0]))

    target_index = 0

    most_similar_index, similarities = (
        processor.embedding_service.most_similar(
            embeddings,
            target_index=target_index
        )
    )

    print("\n========== MOST SIMILAR CHUNK ==========\n")

    print("TARGET CHUNK:", target_index)
    print(
        "TARGET PAGE:",
        chunk_data[target_index]["page_number"]
    )

    print(
        "TARGET TEXT:\n",
        chunk_data[target_index]["text"][:500]
    )

    print("\nMOST SIMILAR CHUNK:", most_similar_index)

    print(
        "MOST SIMILAR PAGE:",
        chunk_data[most_similar_index]["page_number"]
    )

    print(
        "SIMILARITY:",
        round(float(similarities[most_similar_index]), 4)
    )

    print(
        "MOST SIMILAR TEXT:\n",
        chunk_data[most_similar_index]["text"][:500]
    )

    return {
        "status": "success",
        "pages": extracted_data["page_count"],
        "image": extracted_image,
        "chunks": chunk_data,
        "structure": {
            "document_id": structure["document_id"],
            "pages": structure["page_count"],
            "blocks": structure["blocks"],
        },
    }