"""
FastAPI Service for Graph-RAG System
Handles document processing and question answering
"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import redis
import json
import os
from dotenv import load_dotenv
import logging
from fastapi.responses import JSONResponse

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Graph-RAG Processing Service",
    description="Document processing and Q&A service for Graph-RAG system",
    version="1.0.0"
)

# تنظیم CORS برای ارتباط با Laravel
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # در تولید، دامنه‌های خاص را مشخص کنید
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Redis Connection ====================
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5
)

# تست اتصال به Redis
try:
    redis_client.ping()
    logger.info("✅ Connected to Redis successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")

# ==================== Initialize RAG Service ====================
RAG_AVAILABLE = False
rag_service = None

try:
    from app.services.agentic_rag_service import AgenticRAGService as RAGService
    rag_service = RAGService()
    RAG_AVAILABLE = True
    logger.info("✅ RAG Service initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG Service: {e}")

# ==================== Pydantic Models ====================

class DocumentUploadEvent(BaseModel):
    """مدل برای درخواست پردازش سند"""
    user_id: int
    session_id: int
    file_path: str
    document_ids: List[int]

class QuestionRequest(BaseModel):
    """مدل برای درخواست سوال"""
    question: str
    document_ids: Optional[List[int]] = None
    top_k: Optional[int] = 5
    session_id: Optional[int] = None

class QuestionResponse(BaseModel):
    """مدل برای پاسخ سوال"""
    answer: str
    sources: List[Dict[str, Any]]
    question: str
    num_chunks: int
    model: str
    document_ids: Optional[List[int]] = None

class HealthResponse(BaseModel):
    """مدل برای بررسی سلامت"""
    status: str
    redis: bool
    rag_service: bool
    model: Optional[str] = None

# ==================== Endpoints ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """صفحه اصلی"""
    return {
        "message": "Graph-RAG Processing Service",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/process": "Process documents (POST)",
            "/ask": "Ask questions (POST)",
            "/ask-simple": "Ask simple question (POST)",
            "/status": "Get processing status (GET)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """بررسی سلامت سرویس"""
    redis_status = False
    try:
        redis_client.ping()
        redis_status = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if redis_status and RAG_AVAILABLE else "degraded",
        redis=redis_status,
        rag_service=RAG_AVAILABLE,
        model=rag_service.model_name if RAG_AVAILABLE else None
    )

@app.post("/process")
async def process_document(event: DocumentUploadEvent, background_tasks: BackgroundTasks):
    """
    پردازش اسناد آپلود شده
    
    - فایل را از Redis Queue به Worker می‌فرستد
    - پردازش در پس‌زمینه انجام می‌شود
    """
    try:
        # اضافه کردن به صف Redis
        redis_client.rpush(
            'document_queue',
            json.dumps(event.dict())
        )
        
        logger.info(f"📄 Document queued for processing: {event.file_path}")
        
        return {
            "status": "queued",
            "message": "Document processing started",
            "user_id": event.user_id,
            "session_id": event.session_id,
            "document_ids": event.document_ids,
            "queue_position": redis_client.llen('document_queue')
        }
        
    except Exception as e:
        logger.error(f"❌ Error queueing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    پرسش سوال از سیستم RAG
    
    - سوال را به امبدینگ تبدیل می‌کند
    - در Qdrant جستجو می‌کند
    - با Ollama پاسخ می‌دهد
    """
    if not RAG_AVAILABLE or not rag_service:
        raise HTTPException(
            status_code=503, 
            detail="RAG Service not available. Please check Ollama and Qdrant."
        )
    
    try:
        logger.info(f"🤔 Question: {request.question}")
        logger.info(f"   Document IDs: {request.document_ids}")
        
        # استفاده از متد `answer` (نه `ask_question`)
        result = rag_service.answer(
            question=request.question,
            document_ids=request.document_ids
        )
        
        return QuestionResponse(
            answer=result.get('answer', 'No answer generated'),
            sources=result.get('sources', []),
            question=request.question,
            num_chunks=result.get('num_searches', 0),  # توجه: num_searches
            model=rag_service.model_name,
            document_ids=request.document_ids
        )
        
    except Exception as e:
        logger.error(f"❌ Error asking question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask-simple")
async def ask_simple_question(question: str):
    """
    پرسش سوال ساده بدون جستجو در اسناد
    
    - فقط از Ollama استفاده می‌کند
    - برای تست سریع مفید است
    """
    if not RAG_AVAILABLE or not rag_service:
        raise HTTPException(
            status_code=503, 
            detail="RAG Service not available"
        )
    
    try:
        import ollama
        response = ollama.generate(
            model=rag_service.model_name,
            prompt=question
        )
        
        return {
            "question": question,
            "answer": response['response'],
            "model": rag_service.model_name
        }
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{document_id}")
async def get_document_status(document_id: int):
    """
    دریافت وضعیت پردازش یک سند
    
    - از Redis می‌خواند
    - وضعیت: pending, processing, completed, failed
    """
    try:
        # بررسی در Redis
        key = f"document:status:{document_id}"
        status = redis_client.get(key)
        
        if status:
            return {
                "document_id": document_id,
                "status": json.loads(status)
            }
        else:
            return {
                "document_id": document_id,
                "status": "not_found",
                "message": "Document not found or not processed yet"
            }
            
    except Exception as e:
        logger.error(f"❌ Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queue/status")
async def get_queue_status():
    """دریافت وضعیت صف پردازش"""
    try:
        queue_length = redis_client.llen('document_queue')
        
        return {
            "queue_length": queue_length,
            "is_processing": queue_length > 0,
            "max_queue_size": 1000
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Webhook Endpoint ====================

class WebhookData(BaseModel):
    document_id: int
    status: str  # success, failed
    pages: Optional[int] = None
    chunks: Optional[int] = None
    entities: Optional[int] = None
    figures: Optional[int] = None
    tables: Optional[int] = None
    qdrant_points: Optional[int] = None
    error: Optional[str] = None

@app.post("/webhook")
async def webhook_handler(data: WebhookData):
    """
    دریافت Webhook از Worker
    
    - Worker بعد از اتمام پردازش، نتیجه را به اینجا می‌فرستد
    - وضعیت را در Redis ذخیره می‌کند
    """
    try:
        logger.info(f"📨 Webhook received for document {data.document_id}")
        logger.info(f"   Status: {data.status}")
        
        # ذخیره در Redis
        key = f"document:status:{data.document_id}"
        redis_client.setex(
            key,
            3600,  # 1 ساعت
            json.dumps(data.dict())
        )
        
        return {
            "status": "received",
            "document_id": data.document_id,
            "message": "Webhook processed successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    )

# ==================== Startup Events ====================

@app.on_event("startup")
async def startup_event():
    """رویداد شروع سرویس"""
    logger.info("🚀 Starting Graph-RAG Processing Service...")
    logger.info(f"   Redis: {os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', 6379)}")
    logger.info(f"   RAG Service: {'Available' if RAG_AVAILABLE else 'Not available'}")
    
    if RAG_AVAILABLE and rag_service:
        logger.info(f"   Model: {rag_service.model_name}")

@app.on_event("shutdown")
async def shutdown_event():
    """رویداد توقف سرویس"""
    logger.info("🛑 Shutting down Graph-RAG Processing Service...")

# ==================== Run the App ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )