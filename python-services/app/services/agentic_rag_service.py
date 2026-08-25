# app/services/agentic_rag_service.py

import ollama
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import json
import re
import logging

logger = logging.getLogger(__name__)

class AgenticRAGService:
    """سرویس Agentic RAG با قابلیت شکستن سوال و جستجوی چندمرحله‌ای"""
    
    def __init__(self, model_name: str = "gemma3:4b"):
        print(f"🔧 Initializing Agentic RAG Service with model: {model_name}")
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.collection_name = "documents"
        self.max_iterations = 3
        
        # تست اتصال به Ollama
        try:
            ollama.list()
            print("   ✅ Connected to Ollama successfully")
        except Exception as e:
            print(f"   ❌ Failed to connect to Ollama: {e}")
            raise
    
    def ask_question(self, question: str, document_ids: Optional[List[int]] = None, top_k: int = 5) -> Dict[str, Any]:
        """Wrapper برای سازگاری با main.py - مستقیماً به answer می‌دهد"""
        return self.answer(question, document_ids)
    
    def answer(self, question: str, document_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        پاسخگویی هوشمند با شکستن سوال و جستجوی چندمرحله‌ای
        """
        print(f"\n🤔 Original Question: {question}")
        
        # مرحله ۱: تحلیل سوال و شکستن به زیرسوالات
        sub_queries = self._decompose_question(question)
        print(f"📋 Sub-queries: {sub_queries}")
        
        # مرحله ۲: تعیین نوع جستجو برای هر زیرسوال
        search_plans = []
        for sub_q in sub_queries:
            plan = self._decide_search_type(sub_q)
            search_plans.append(plan)
        print(f"🎯 Search plans: {search_plans}")
        
        # مرحله ۳: اجرای جستجوها
        all_results = []
        for i, (sub_q, plan) in enumerate(zip(sub_queries, search_plans)):
            print(f"\n🔍 Executing search {i+1}: {sub_q}")
            
            if plan == 'vector':
                results = self._vector_search(sub_q, document_ids, top_k=3)
            elif plan == 'graph':
                results = self._graph_search(sub_q, document_ids)
            else:  # hybrid
                vector_results = self._vector_search(sub_q, document_ids, top_k=2)
                graph_results = self._graph_search(sub_q, document_ids)
                results = self._merge_results(vector_results, graph_results)
            
            all_results.append({
                'sub_query': sub_q,
                'search_type': plan,
                'results': results
            })
        
        # مرحله ۴: ترکیب و سنتز پاسخ
        final_answer = self._synthesize_answer(question, all_results)
        
        # مرحله ۵: استخراج منابع
        sources = self._extract_sources(all_results)
        
        return {
            'answer': final_answer,
            'sub_queries': sub_queries,
            'search_plans': search_plans,
            'sources': sources,
            'num_searches': len(sub_queries)
        }
    
    def _decompose_question(self, question: str) -> List[str]:
        """شکستن سوال به زیرسوالات با استفاده از LLM"""
        prompt = f"""You are a query decomposition expert. Break down the following complex question into 2-3 simpler sub-questions that can be answered independently.

Question: {question}

Return ONLY the sub-questions as a JSON array, e.g.:
["Sub-question 1", "Sub-question 2", "Sub-question 3"]

Sub-questions:"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.1}
            )
            
            # استخراج JSON از پاسخ
            text = response['response'].strip()
            # پیدا کردن آرایه JSON
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                sub_queries = json.loads(match.group())
                return sub_queries
            else:
                # اگر JSON پیدا نشد، خود سوال را برگردان
                return [question]
                
        except Exception as e:
            print(f"⚠️ Decomposition error: {e}")
            return [question]
    
    def _decide_search_type(self, sub_query: str) -> str:
        """تصمیم‌گیری درباره نوع جستجو (بردار، گراف، یا ترکیبی)"""
        # کلمات کلیدی برای جستجوی گراف
        graph_keywords = ['relationship', 'compare', 'difference', 'similar', 'citation', 'reference', 'author']
        # کلمات کلیدی برای جستجوی بردار
        vector_keywords = ['definition', 'meaning', 'what is', 'explain', 'describe']
        
        sub_query_lower = sub_query.lower()
        
        # اگر کلمات گراف وجود داشت، از گراف استفاده کن
        if any(kw in sub_query_lower for kw in graph_keywords):
            return 'graph'
        # اگر کلمات بردار وجود داشت، از بردار استفاده کن
        elif any(kw in sub_query_lower for kw in vector_keywords):
            return 'vector'
        # در غیر این صورت ترکیبی
        else:
            return 'hybrid'
    
    def _vector_search(self, query: str, document_ids: Optional[List[int]], top_k: int = 3) -> List[Dict]:
        """جستجوی برداری در Qdrant"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        search_filter = None
        if document_ids:
            search_filter = {
                "must": [
                    {"key": "doc_id", "match": {"value": doc_id}}
                    for doc_id in document_ids
                ]
            }
        
        # استفاده از query_points به جای search
        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=search_filter
        )
        
        return [
            {
                'text': r.payload.get('text', ''),
                'score': r.score,
                'doc_id': r.payload.get('doc_id'),
                'chunk_index': r.payload.get('chunk_index'),
                'type': 'vector'
            }
            for r in results.points
        ]
    
    def _graph_search(self, query: str, document_ids: Optional[List[int]]) -> List[Dict]:
        """جستجوی گراف در Neo4j"""
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        
        results = []
        
        try:
            with driver.session() as session:
                cypher_query = """
                MATCH (d:Document)-[:CONTAINS]->(e:Entity)
                WHERE d.id IN $doc_ids OR $doc_ids IS NULL
                AND e.name CONTAINS $query
                RETURN e.name, e.type, d.id as doc_id
                LIMIT 10
                """
                result = session.run(cypher_query, {
                    'doc_ids': document_ids if document_ids else [],
                    'query': query
                })
                
                for record in result:
                    results.append({
                        'text': f"{record['e.name']} ({record['e.type']})",
                        'score': 1.0,
                        'doc_id': record['doc_id'],
                        'type': 'graph',
                        'entity_name': record['e.name'],
                        'entity_type': record['e.type']
                    })
                    
        except Exception as e:
            print(f"⚠️ Graph search error: {e}")
        finally:
            driver.close()
        
        return results
    
    def _merge_results(self, vector_results: List[Dict], graph_results: List[Dict]) -> List[Dict]:
        """ادغام نتایج برداری و گراف"""
        merged = vector_results + graph_results
        
        # حذف تکراری‌ها بر اساس متن
        seen = set()
        unique_results = []
        for r in merged:
            text_key = r['text'][:100]
            if text_key not in seen:
                seen.add(text_key)
                unique_results.append(r)
        
        return unique_results
    
    def _synthesize_answer(self, question: str, all_results: List[Dict]) -> str:
        """ترکیب و سنتز پاسخ نهایی"""
        context = ""
        for i, item in enumerate(all_results):
            context += f"\n--- Search {i+1}: {item['sub_query']} ---\n"
            for result in item['results'][:3]:
                context += f"- {result['text'][:300]}\n"
        
        prompt = f"""You are a research assistant synthesizing answers from multiple searches.

Original Question: {question}

Search Results:
{context}

Instructions:
1. Answer the original question based on ALL the search results
2. If you find conflicting information, mention both perspectives
3. Cite which search result each piece of information comes from
4. If the information is insufficient, say so clearly

Answer:"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={'temperature': 0.3}
            )
            return response['response'].strip()
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def _extract_sources(self, all_results: List[Dict]) -> List[Dict]:
        """استخراج منابع از نتایج"""
        sources = []
        seen = set()
        
        for item in all_results:
            for result in item['results']:
                doc_id = result.get('doc_id')
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    sources.append({
                        'document_id': doc_id,
                        'search_type': result.get('type', 'unknown'),
                        'text': result.get('text', '')[:200]
                    })
        
        return sources