"""
سرویس تولید و ذخیره امبدینگ‌ها در Qdrant
"""

import uuid
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """سرویس مدیریت امبدینگ‌ها و Qdrant"""
    
    def __init__(self):
        """مقداردهی اولیه"""
        print("   🔧 Initializing EmbeddingService...")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "documents"
        self.vector_size = 384  # اندازه خروجی مدل
        self._create_collection()
        
    def _create_collection(self):
        """ایجاد کالکشن در Qdrant اگر وجود نداشته باشد"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"   ✅ Collection '{self.collection_name}' created in Qdrant")
            else:
                print(f"   ✅ Collection '{self.collection_name}' already exists")
                
        except Exception as e:
            print(f"   ❌ Error creating collection: {e}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """تولید امبدینگ برای یک متن"""
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"   ❌ Error generating embedding: {e}")
            return [0.0] * self.vector_size
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """تولید امبدینگ برای چند متن به صورت دسته‌ای"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"   ❌ Error generating batch embeddings: {e}")
            return [[0.0] * self.vector_size for _ in texts]
    
    def store_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """ذخیره یک چانک در Qdrant"""
        try:
            point_id = str(uuid.uuid4())
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    'chunk_id': chunk_data.get('chunk_id'),
                    'doc_id': chunk_data.get('doc_id'),
                    'text': chunk_data.get('text'),
                    'chunk_index': chunk_data.get('metadata', {}).get('chunk_index'),
                    'total_chunks': chunk_data.get('metadata', {}).get('total_chunks'),
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error storing chunk: {e}")
            return False
    
    def store_chunks_batch(self, chunks: List[Dict[str, Any]]) -> bool:
        """ذخیره چند چانک به صورت دسته‌ای"""
        try:
            points = []
            for chunk in chunks:
                embedding = chunk.get('embedding')
                if embedding is None:
                    # اگر امبدینگ وجود ندارد، تولید کن
                    embedding = self.generate_embedding(chunk.get('text', ''))
                
                point_id = str(uuid.uuid4())
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        'chunk_id': chunk.get('chunk_id'),
                        'doc_id': chunk.get('doc_id'),
                        'text': chunk.get('text'),
                        'chunk_index': chunk.get('metadata', {}).get('chunk_index'),
                        'total_chunks': chunk.get('metadata', {}).get('total_chunks'),
                    }
                )
                points.append(point)
            
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"   ✅ Stored {len(points)} chunks in Qdrant")
                return True
                
        except Exception as e:
            print(f"   ❌ Error storing chunks: {e}")
            return False
    
    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """جستجوی مشابهت در Qdrant"""
        try:
            query_embedding = self.generate_embedding(query)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            return [
                {
                    'score': result.score,
                    'text': result.payload.get('text', ''),
                    'chunk_id': result.payload.get('chunk_id'),
                    'doc_id': result.payload.get('doc_id'),
                    'chunk_index': result.payload.get('chunk_index'),
                }
                for result in results
            ]
            
        except Exception as e:
            print(f"   ❌ Error searching: {e}")
            return []
    
    def get_collection_info(self) -> Dict:
        """دریافت اطلاعات کالکشن"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size
            }
        except Exception as e:
            print(f"   ❌ Error getting collection info: {e}")
            return {}