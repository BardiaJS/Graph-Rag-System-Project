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
    processor = DocumentProcessor(request.file_path , request.user_id, request.document_ids[0])
    extracted_data = processor.text_extraction()
    extracted_image = processor.image_extraction()
    chunk_data = processor.chunk_service(extracted_data["pages"])
    structure = processor.structure_extraction()
    
    print("\n========== OVERLAP DEBUG ==========")

    for i in range(3):
        current = chunk_data[i]["text"]
        next_chunk = chunk_data[i + 1]["text"]

        print(f"\n--- Chunk {i} END ---")
        print(current[-150:])

        print(f"\n--- Chunk {i + 1} START ---")
        print(next_chunk[:150])

    print("====================================\n")

    print(extracted_data["pages"][0]["text"])


    print("\n========== STRUCTURE DEBUG ==========")

    for page in extracted_data["pages"]:
        print(f"\n========== PAGE {page['page_number']} ==========")
        print(page["text"][:1500])

    print("======================================\n")

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


        evaluator = ChunkEvaluator(chunk_data)

        evaluation = evaluator.evaluate()

        print("\n========== CHUNK EVALUATION ==========\n")

        for key, value in evaluation.items():
            print(f"{key}: {value}")


        boundaries = evaluator.evaluate_boundaries()

        summary = evaluator.summarize_boundaries(boundaries)

        print("\n========== BOUNDARY SUMMARY ==========\n")

        for key, value in summary.items():
            print(f"{key}: {value}")


        print("\n========== SUSPICIOUS BOUNDARIES ==========\n")

        for boundary in boundaries:

            if boundary["category"] in {
                "WORD_SPLIT",
                "SUSPICIOUS",
            }:

                print(
                    f"\n"
                    f"{boundary['current_chunk']} -> "
                    f"{boundary['next_chunk']}"
                )

                print(
                    f"PAGE: "
                    f"{boundary['current_page']} -> "
                    f"{boundary['next_page']}"
                )

                print(
                    f"CATEGORY: "
                    f"{boundary['category']}"
                )

                print(
                    f"OVERLAP: "
                    f"{boundary['overlap']}"
                )

                print(
                    f"CURRENT END: "
                    f"{boundary['current_end']}"
                )

                print(
                    f"NEXT START: "
                    f"{boundary['next_start']}"
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
        }
    }




    

