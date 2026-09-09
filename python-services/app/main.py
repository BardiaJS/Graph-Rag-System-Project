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
        request.document_ids[0]
    )

    structure = processor.structure_extraction()

    print("\n========== NORMALIZED STRUCTURE ==========\n")

    for block in structure["blocks"][:40]:

        print(
            f"PAGE: {block['page_number']} | "
            f"TYPE: {block['content_type']} | "
            f"HEADER LEVEL: {block['header_level']}"
        )

        if block["text"]:
            print(block["text"][:300])

        if block["content_type"] == "picture":
            print("IMAGE:", block["image"])

        if block["content_type"] == "table":
            print("TABLE:", block["table"])

        print("-" * 80)

    return {
        "status": "success",
        "document_id": structure["document_id"],
        "pages": structure["page_count"],
        "blocks": structure["blocks"],
    }


    

