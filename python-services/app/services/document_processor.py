"""
Document processor service for Graph-RAG system
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
import logging
import redis.asyncio as redis
from datetime import datetime
import httpx  # اضافه کنید در بالای فایل

# تنظیم logger
logger = logging.getLogger(__name__)

# ایمپورت سرویس‌های مورد نیاز
try:
    from app.services.embedding_service import EmbeddingService
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️ EmbeddingService not available. Install: pip install sentence-transformers qdrant-client")

try:
    from app.services.neo4j_service import Neo4jService
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4jService not available. Install: pip install neo4j")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("⚠️ PyMuPDF not available. Install: pip install PyMuPDF")


class DocumentProcessor:
    """پردازشگر اصلی اسناد برای استخراج متن، ساخت گراف و امبدینگ"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        """مقداردهی اولیه پردازشگر"""
        print("🔧 Initializing DocumentProcessor...")
        self.storage_path = Path(__file__).parent.parent.parent.parent / 'backend/storage/app/public'
        print(f"   📁 Storage path: {self.storage_path}")
        
        # مقداردهی Redis
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=0,
                decode_responses=True
            )
            print("   ✅ Redis client initialized")
        except Exception as e:
            print(f"   ⚠️ Redis initialization failed: {e}")
            self.redis_client = None
        
        # مقداردهی Neo4j
        if NEO4J_AVAILABLE:
            try:
                self.neo4j_service = Neo4jService()
                print("   ✅ Neo4jService initialized")
            except Exception as e:
                print(f"   ⚠️ Neo4jService initialization failed: {e}")
                self.neo4j_service = None
        else:
            self.neo4j_service = None
        
        # مقداردهی EmbeddingService
        if EMBEDDING_AVAILABLE:
            try:
                self.embedding_service = EmbeddingService()
                print("   ✅ EmbeddingService initialized")
            except Exception as e:
                print(f"   ⚠️ EmbeddingService initialization failed: {e}")
                self.embedding_service = None
        else:
            self.embedding_service = None
    
    # ===================== متدهای اصلی =====================
    
    async def _extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """استخراج متن و تصاویر از PDF"""
        if not PYMUPDF_AVAILABLE:
            return {
                'pages': 0,
                'full_text': '',
                'figures': [],
                'tables': [],
                'title': 'Unknown'
            }
        
        doc = None
        try:
            print(f"   📖 Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            print(f"   📄 Number of pages: {page_count}")
            
            full_text = ""
            figures = []
            tables = []
            title = pdf_path.stem
            
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text()
                full_text += text + "\n\n"
                
                # استخراج تصاویر
                images = page.get_images(full=True)
                for img in images:
                    figures.append({
                        'page': page_num + 1,
                        'index': len(figures) + 1
                    })
            
            print(f"   📝 Extracted {len(figures)} figures")
            print(f"   📝 Full text length: {len(full_text)} characters")
            
            return {
                'pages': page_count,
                'full_text': full_text,
                'figures': figures,
                'tables': tables,
                'title': title
            }
            
        except Exception as e:
            print(f"   ⚠️ Error extracting text: {e}")
            import traceback
            traceback.print_exc()
            return {
                'pages': 0,
                'full_text': '',
                'figures': [],
                'tables': [],
                'title': 'Unknown'
            }
        finally:
            if doc:
                doc.close()
    
    async def _semantic_chunking(self, text: str) -> List[Dict]:
        """تقسیم‌بندی معنایی متن"""
        if not text:
            return []
        
        # تقسیم‌بندی بر اساس پاراگراف
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                # حذف فضاهای اضافی و یکپارچه‌سازی
                clean_text = ' '.join(para.strip().split())
                if len(clean_text) > 50:  # فقط پاراگراف‌های با طول کافی
                    chunks.append({
                        'chunk_id': f'chunk_{i}',
                        'text': clean_text,
                        'metadata': {
                            'chunk_index': i,
                            'total_chunks': len(paragraphs),
                            'char_count': len(clean_text),
                            'token_count': len(clean_text.split())
                        }
                    })
        
        # اگر تعداد چانک‌ها خیلی کم است، به بخش‌های کوچکتر تقسیم کن
        if len(chunks) < 3 and len(text) > 1000:
            sentences = text.split('. ')
            temp_chunks = []
            current_chunk = ""
            
            for sent in sentences:
                if len(current_chunk) + len(sent) < 1000:
                    current_chunk += sent + ". "
                else:
                    if current_chunk.strip():
                        temp_chunks.append({
                            'chunk_id': f'chunk_{len(temp_chunks)}',
                            'text': current_chunk.strip(),
                            'metadata': {
                                'chunk_index': len(temp_chunks),
                                'total_chunks': 0,
                                'char_count': len(current_chunk),
                                'token_count': len(current_chunk.split())
                            }
                        })
                    current_chunk = sent + ". "
            
            if current_chunk.strip():
                temp_chunks.append({
                    'chunk_id': f'chunk_{len(temp_chunks)}',
                    'text': current_chunk.strip(),
                    'metadata': {
                        'chunk_index': len(temp_chunks),
                        'total_chunks': 0,
                        'char_count': len(current_chunk),
                        'token_count': len(current_chunk.split())
                    }
                })
            
            chunks = temp_chunks
        
        # بروزرسانی total_chunks
        for i, chunk in enumerate(chunks):
            chunk['metadata']['total_chunks'] = len(chunks)
            chunk['metadata']['chunk_index'] = i
            chunk['doc_id'] = None
        
        return chunks
    
    async def _extract_entities_from_text(self, text: str) -> List[Dict]:
        """استخراج موجودیت‌ها از متن با الگوهای ساده"""
        if not text:
            return []
        
        entities = []
        
        # الگوهای تشخیص موجودیت
        patterns = {
            'method': r'\b(method|approach|technique|algorithm|framework|model|architecture|network|system|procedure|protocol)\b',
            'dataset': r'\b(dataset|data|benchmark|corpus|collection|database|knowledge[ -]base|repository)\b',
            'metric': r'\b(accuracy|precision|recall|f1[ -_]score|auc|roc|error|loss|perplexity|bleu|rouge|mse|mae|rmse)\b',
            'technology': r'\b(Python|TensorFlow|PyTorch|Keras|scikit[ -]learn|Transformers|GPT|BERT|LLM|NLP|CNN|RNN|LSTM|GAN|VAE|MLP|SVM|XGBoost|LightGBM|CatBoost)\b',
            'concept': r'\b(machine learning|deep learning|artificial intelligence|neural network|computer vision|natural language processing|reinforcement learning|transfer learning|federated learning)\b',
            'organization': r'\b(Google|Microsoft|Amazon|Facebook|OpenAI|DeepMind|Meta|IBM|Intel|NVIDIA|AMD|Apple|Stanford|MIT|Berkeley|Oxford|Cambridge)\b'
        }
        
        # جستجو در متن
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match_lower = match.lower()
                if not any(e['name'].lower() == match_lower for e in entities):
                    entities.append({
                        'name': match,
                        'type': entity_type,
                        'description': f'Extracted from document as {entity_type}'
                    })
        
        return entities
    
    async def _save_to_neo4j(
        self, 
        doc_id: int, 
        title: str, 
        user_id: int, 
        session_id: int,
        entities: List[Dict],
        chunks: List[Dict]
    ):
        """ذخیره در Neo4j"""
        if not self.neo4j_service:
            print(f"   ⚠️ Neo4jService not available, skipping save for document {doc_id}")
            return
        
        try:
            # ذخیره سند
            success = self.neo4j_service.save_document(
                doc_id=doc_id,
                title=title,
                user_id=user_id,
                session_id=session_id
            )
            
            if success:
                print(f"   ✅ Document {doc_id} saved to Neo4j")
            else:
                print(f"   ⚠️ Failed to save document {doc_id} to Neo4j")
            
            # ذخیره موجودیت‌ها
            if entities:
                success = self.neo4j_service.save_entities(doc_id, entities)
                if success:
                    print(f"   ✅ Saved {len(entities)} entities for document {doc_id}")
                else:
                    print(f"   ⚠️ Failed to save entities for document {doc_id}")
            
        except Exception as e:
            print(f"   ❌ Error saving to Neo4j: {e}")
            import traceback
            traceback.print_exc()
    
    async def _save_result_to_redis(self, result: dict, doc_id: int):
        """ذخیره نتیجه پردازش در Redis"""
        try:
            if not hasattr(self, 'redis_client') or self.redis_client is None:
                logger.warning(f"Redis client not available, saving to file for doc {doc_id}")
                fallback_dir = Path('/tmp/redis_fallback')
                fallback_dir.mkdir(exist_ok=True)
                filename = fallback_dir / f'doc_{doc_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                with open(filename, 'w') as f:
                    json.dump(result, f, default=str, indent=2)
                logger.info(f"✅ Result saved to file: {filename}")
                return
            
            key = f"document:process:{doc_id}"
            result['saved_at'] = datetime.now().isoformat()
            await self.redis_client.setex(
                key,
                86400,
                json.dumps(result, default=str)
            )
            logger.info(f"✅ Result saved to Redis for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to save result to Redis for doc {doc_id}: {e}")

    async def _send_result_to_laravel(self, result: dict, document_ids: List[int]):
        """ارسال نتیجه به Laravel API"""
        try:
            url = "http://localhost:8000/api/webhook/document-processed"
            
            for doc_id in document_ids:
                data = {
                    'document_id': doc_id,
                    'status': result.get('status', 'success'),
                    'pages': result.get('pages', 0),
                    'chunks': result.get('chunks', 0),
                    'entities': result.get('entities', 0),
                    'figures': result.get('figures', 0),
                    'tables': result.get('tables', 0),
                    'qdrant_points': result.get('qdrant_points', 0),
                    'file_size': result.get('file_size', 0)
                }
                
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.post(url, json=data)
                    
                    if response.status_code == 200:
                        print(f"   ✅ Result sent to Laravel for document {doc_id}")
                        logger.info(f"Webhook sent for doc {doc_id}: {response.json()}")
                    else:
                        print(f"   ⚠️ Webhook failed for doc {doc_id}: {response.status_code}")
                        logger.warning(f"Webhook failed for doc {doc_id}: {response.text}")
                        
        except Exception as e:
            logger.error(f"Error sending webhook: {e}")
            print(f"   ❌ Webhook error: {e}")

    # ===================== متد اصلی پردازش =====================
    
    async def process(
        self,
        file_path: str,
        user_id: int,
        session_id: int,
        document_ids: List[int]
    ) -> Dict[str, Any]:
        """پردازش یک سند"""
        print(f"\n   📖 Starting document processing...")
        print(f"   📄 File path: {file_path}")
        print(f"   👤 User ID: {user_id}")
        print(f"   💬 Session ID: {session_id}")
        print(f"   📋 Document IDs: {document_ids}")
        
        try:
            full_path = self.storage_path / file_path
            
            if not full_path.exists():
                print(f"   ⚠️ File not found: {full_path}")
                return {
                    'status': 'failed',
                    'message': f'File not found: {file_path}',
                    'document_ids': document_ids,
                    'error': 'FILE_NOT_FOUND'
                }
            
            print(f"   ✅ File found at: {full_path}")
            file_size = full_path.stat().st_size
            print(f"   📊 File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
            
            # مرحله ۱: استخراج متن
            print("\n   📝 Step 1: Extracting text from PDF...")
            extracted_data = await self._extract_text_from_pdf(full_path)
            
            print(f"   ✅ Extracted {extracted_data.get('pages', 0)} pages")
            print(f"   🖼️ Found {len(extracted_data.get('figures', []))} figures")
            print(f"   📊 Found {len(extracted_data.get('tables', []))} tables")
            
            # مرحله ۲: تقسیم‌بندی
            print("\n   ✂️ Step 2: Semantic chunking...")
            chunks = await self._semantic_chunking(extracted_data.get('full_text', ''))
            print(f"   ✅ Created {len(chunks)} semantic chunks")
            
            # مرحله ۳: ساخت گراف
            print("\n   🕸️ Step 3: Building knowledge graph...")
            entities = await self._extract_entities_from_text(extracted_data.get('full_text', ''))
            print(f"   ✅ Extracted {len(entities)} entities")
            
            # مرحله ۴: تولید و ذخیره امبدینگ‌ها
            print("\n   🔢 Step 4: Generating and storing embeddings...")
            qdrant_points = 0
            
            if self.embedding_service and chunks:
                try:
                    texts = [chunk.get('text', '') for chunk in chunks]
                    embeddings = self.embedding_service.generate_embeddings_batch(texts)
                    
                    for i, chunk in enumerate(chunks):
                        chunk['embedding'] = embeddings[i] if i < len(embeddings) else None
                    
                    success = self.embedding_service.store_chunks_batch(chunks)
                    
                    if success:
                        print(f"   ✅ Stored {len(chunks)} chunks in Qdrant")
                        collection_info = self.embedding_service.get_collection_info()
                        qdrant_points = collection_info.get('points_count', 0)
                        print(f"   📊 Qdrant collection: {qdrant_points} points")
                    else:
                        print(f"   ⚠️ Failed to store chunks in Qdrant")
                        
                except Exception as e:
                    print(f"   ❌ Error in embedding service: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if not self.embedding_service:
                    print("   ⚠️ EmbeddingService not available, skipping Qdrant storage")
                else:
                    print("   ⚠️ No chunks to store in Qdrant")
            
            # مرحله ۵: ذخیره در Neo4j
            print("\n   💾 Step 5: Saving to Neo4j...")
            
            for doc_id in document_ids:
                try:
                    await self._save_to_neo4j(
                        doc_id=doc_id,
                        title=extracted_data.get('title', f'Document_{doc_id}'),
                        user_id=user_id,
                        session_id=session_id,
                        entities=entities,
                        chunks=chunks
                    )
                except Exception as e:
                    print(f"   ❌ Error saving document {doc_id} to Neo4j: {e}")
            
            # نتیجه نهایی
            result = {
                'status': 'success',
                'pages': extracted_data.get('pages', 0),
                'chunks': len(chunks),
                'entities': len(entities),
                'figures': len(extracted_data.get('figures', [])),
                'tables': len(extracted_data.get('tables', [])),
                'qdrant_points': qdrant_points,
                'file_size': file_size,
                'document_ids': document_ids,
                'user_id': user_id,
                'session_id': session_id
            }
            
            # ذخیره در Redis
            for doc_id in document_ids:
                try:
                    await self._save_result_to_redis(result, doc_id)
                except Exception as e:
                    logger.error(f"Error in Redis save for doc {doc_id}: {e}")
            
            # ارسال به Laravel
            try:
                await self._send_result_to_laravel(result, document_ids)
            except Exception as e:
                logger.error(f"Laravel API error: {e}")
            
            print(f"   ✅ Document processed successfully!")
            return result
            
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'message': f'Processing failed: {str(e)}',
                'document_ids': document_ids,
                'error': 'PROCESSING_ERROR'
            }


# تست
if __name__ == "__main__":
    print("🧪 Testing DocumentProcessor...")
    
    async def test():
        processor = DocumentProcessor()
        result = await processor.process(
            file_path='documents/1/test.pdf',
            user_id=1,
            session_id=1,
            document_ids=[1]
        )
        print(f"\n📋 Test Result: {json.dumps(result, indent=2)}")
    
    asyncio.run(test())