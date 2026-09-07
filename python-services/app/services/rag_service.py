from app.services.agentic_rag_service import (
    AgenticRAGService,
)


class RAGService(AgenticRAGService):
    """
    Compatibility wrapper برای کدهای قبلی.
    """

    def __init__(
        self,
        model_name: str = "gemma3:4b",
    ):

        print(
            "🔧 Initializing RAGService "
            "(wrapper for AgenticRAGService)"
        )

        super().__init__(
            model_name=model_name
        )