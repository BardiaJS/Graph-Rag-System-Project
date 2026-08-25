# app/services/rag_service.py
from app.services.agentic_rag_service import AgenticRAGService

class RAGService(AgenticRAGService):
    """Wrapper برای سازگاری با main.py"""
    
    def __init__(self, model_name: str = "gemma3:4b"):
        print(f"🔧 Initializing RAGService (wrapper for AgenticRAGService)")
        super().__init__(model_name)