from typing import List, Dict, Any
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import hashlib

class SemanticChunker:
    """تقسیم‌بندی معنایی متن با LlamaIndex"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)
        self.splitter = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            chunk_size=512,
            chunk_overlap=50
        )
    
    def chunk_document(self, text: str, doc_id: int) -> List[Dict]:
        """تقسیم‌بندی سند به تکه‌های معنایی"""
        
        # آماده‌سازی سند برای LlamaIndex
        document = LlamaDocument(text=text)
        
        # تقسیم‌بندی
        nodes = self.splitter.get_nodes_from_documents([document])
        
        chunks = []
        for idx, node in enumerate(nodes):
            chunk_id = hashlib.md5(f"{doc_id}_{idx}_{node.text[:50]}".encode()).hexdigest()
            chunks.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'text': node.text,
                'metadata': {
                    'chunk_index': idx,
                    'total_chunks': len(nodes),
                    'char_count': len(node.text),
                    'token_count': len(node.text.split())
                },
                'embedding': node.embedding if hasattr(node, 'embedding') else None
            })
        
        return chunks