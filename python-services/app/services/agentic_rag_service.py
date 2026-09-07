"""
Agentic RAG Service
-------------------

Responsibilities:

1. Query decomposition
2. Search strategy selection
3. Document-aware vector search
4. Graph search using Neo4j
5. Hybrid search
6. Result merging and deduplication
7. Comparison-aware retrieval
8. Answer synthesis using Ollama
9. Source extraction
10. Qdrant debugging

Designed to work with:

- Ollama
- Qdrant
- Neo4j
- SentenceTransformers / EmbeddingService

Important:

For comparison questions such as:

    "Which article provides better information about ML?"

the service retrieves evidence independently from each document
instead of performing only a global top-k search.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import ollama

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
)

from app.services.embedding_service import EmbeddingService
from app.services.neo4j_service import Neo4jService


logger = logging.getLogger(__name__)


class AgenticRAGService:
    """
    Agentic RAG service.

    Normal question:

        Question
           ↓
        Decomposition
           ↓
        Search planning
           ↓
        Vector / Graph / Hybrid
           ↓
        Merge
           ↓
        Synthesis
           ↓
        Sources


    Comparison question:

        Question
           ↓
        Detect comparison
           ↓
        Identify documents
           ↓
        Search each document independently
           ↓
        Build balanced context
           ↓
        Compare evidence
           ↓
        Synthesis
           ↓
        Sources
    """

    COLLECTION_NAME = "documents"

    DEFAULT_MODEL = "gemma3:4b"

    DEFAULT_TOP_K = 5

    MAX_SUB_QUERIES = 3

    MAX_CONTEXT_RESULTS_PER_QUERY = 5

    MAX_CONTEXT_CHARS_PER_RESULT = 1800

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "password",
    ):
        print(
            f"🔧 Initializing Agentic RAG Service "
            f"with model: {model_name}"
        )

        self.model_name = model_name
        self.collection_name = self.COLLECTION_NAME
        self.max_iterations = 3

        # =====================================================
        # Qdrant
        # =====================================================

        self.qdrant_client: Optional[QdrantClient] = None

        try:
            self.qdrant_client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
            )

            self.qdrant_client.get_collections()

            print(
                "   ✅ Connected to Qdrant successfully"
            )

        except Exception as e:
            logger.exception(
                "Failed to connect to Qdrant"
            )

            print(
                f"   ⚠️ Qdrant unavailable: {e}"
            )

            self.qdrant_client = None

        # =====================================================
        # Embedding Service
        # =====================================================

        self.embedding_service: Optional[
            EmbeddingService
        ] = None

        try:
            self.embedding_service = EmbeddingService(
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
            )

            print(
                "   ✅ EmbeddingService initialized"
            )

        except Exception as e:
            logger.exception(
                "Failed to initialize EmbeddingService"
            )

            print(
                f"   ⚠️ EmbeddingService unavailable: {e}"
            )

            self.embedding_service = None

        # =====================================================
        # Neo4j
        # =====================================================

        self.neo4j_service: Optional[
            Neo4jService
        ] = None

        try:
            self.neo4j_service = Neo4jService(
                uri=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password,
            )

            if self.neo4j_service.driver:
                print(
                    "   ✅ Neo4jService initialized"
                )
            else:
                print(
                    "   ⚠️ Neo4jService initialized "
                    "but connection is unavailable"
                )

        except Exception as e:
            logger.exception(
                "Failed to initialize Neo4jService"
            )

            print(
                f"   ⚠️ Neo4jService unavailable: {e}"
            )

            self.neo4j_service = None

        # =====================================================
        # Ollama
        # =====================================================

        self.ollama_available = False

        try:
            ollama.list()

            self.ollama_available = True

            print(
                "   ✅ Connected to Ollama successfully"
            )

        except Exception as e:
            logger.exception(
                "Failed to connect to Ollama"
            )

            print(
                f"   ⚠️ Ollama unavailable: {e}"
            )

        # =====================================================
        # Final status
        # =====================================================

        print(
            "   ----------------------------------------"
        )

        print(
            f"   Qdrant: "
            f"{'OK' if self.qdrant_client else 'UNAVAILABLE'}"
        )

        print(
            f"   Embeddings: "
            f"{'OK' if self.embedding_service else 'UNAVAILABLE'}"
        )

        print(
            f"   Neo4j: "
            f"{'OK' if self.neo4j_service and self.neo4j_service.driver else 'UNAVAILABLE'}"
        )

        print(
            f"   Ollama: "
            f"{'OK' if self.ollama_available else 'UNAVAILABLE'}"
        )

        print(
            "   ----------------------------------------"
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def ask_question(
        self,
        question: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        """
        Public wrapper used by main.py.
        """

        return self.answer(
            question=question,
            document_ids=document_ids,
            top_k=top_k,
        )

    def answer(
        self,
        question: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:

        question = (question or "").strip()

        if not question:
            return {
                "answer": "Question cannot be empty.",
                "sub_queries": [],
                "search_plans": [],
                "sources": [],
                "num_searches": 0,
            }

        normalized_document_ids = (
            self._normalize_document_ids(
                document_ids
            )
        )

        top_k = max(
            1,
            min(int(top_k), 20),
        )

        print(
            "\n================================================="
        )

        print(
            "🤖 AGENTIC RAG"
        )

        print(
            "================================================="
        )

        print(
            f"Question: {question}"
        )

        print(
            f"Document IDs: {normalized_document_ids}"
        )

        print(
            f"Top K: {top_k}"
        )

        # =====================================================
        # Detect comparison
        # =====================================================

        is_comparison = self._is_comparison_question(
            question
        )

        print(
            f"Comparison question: {is_comparison}"
        )

        # =====================================================
        # Step 1: Decompose
        # =====================================================

        print(
            "\n🧩 Step 1: Decomposing question..."
        )

        if is_comparison:
            sub_queries = self._build_comparison_sub_queries(
                question
            )
        else:
            sub_queries = self._decompose_question(
                question
            )

        if not sub_queries:
            sub_queries = [question]

        print(
            f"   Sub-queries: {sub_queries}"
        )

        # =====================================================
        # Step 2: Search planning
        # =====================================================

        print(
            "\n🎯 Step 2: Creating search plans..."
        )

        search_plans = []

        for sub_query in sub_queries:
            plan = self._decide_search_type(
                sub_query
            )

            search_plans.append(plan)

        print(
            f"   Search plans: {search_plans}"
        )

        # =====================================================
        # Step 3: Execute searches
        # =====================================================

        print(
            "\n🔍 Step 3: Executing searches..."
        )

        all_results: List[Dict[str, Any]] = []

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Comparison questions must search every document
        # independently.
        # -----------------------------------------------------

        if (
            is_comparison
            and normalized_document_ids
            and len(normalized_document_ids) >= 2
        ):

            all_results = (
                self._comparison_search(
                    question=question,
                    sub_queries=sub_queries,
                    search_plans=search_plans,
                    document_ids=normalized_document_ids,
                    top_k=top_k,
                )
            )

        else:

            for index, (
                sub_query,
                plan,
            ) in enumerate(
                zip(
                    sub_queries,
                    search_plans,
                ),
                start=1,
            ):

                print(
                    f"\n   🔎 Search "
                    f"{index}/{len(sub_queries)}"
                )

                print(
                    f"      Query: {sub_query}"
                )

                print(
                    f"      Type: {plan}"
                )

                try:

                    if plan == "vector":

                        results = self._vector_search(
                            query=sub_query,
                            document_ids=normalized_document_ids,
                            top_k=top_k,
                        )

                    elif plan == "graph":

                        results = self._graph_search(
                            query=sub_query,
                            document_ids=normalized_document_ids,
                        )

                    else:

                        vector_results = (
                            self._vector_search(
                                query=sub_query,
                                document_ids=normalized_document_ids,
                                top_k=top_k,
                            )
                        )

                        graph_results = (
                            self._graph_search(
                                query=sub_query,
                                document_ids=normalized_document_ids,
                            )
                        )

                        results = self._merge_results(
                            vector_results,
                            graph_results,
                            limit=top_k,
                        )

                except Exception as e:

                    logger.exception(
                        f"Search failed for query: {sub_query}"
                    )

                    print(
                        f"      ⚠️ Search failed: {e}"
                    )

                    results = []

                print(
                    f"      Results: {len(results)}"
                )

                all_results.append(
                    {
                        "sub_query": sub_query,
                        "search_type": plan,
                        "results": results,
                    }
                )

        # =====================================================
        # Step 4: Synthesis
        # =====================================================

        print(
            "\n🧠 Step 4: Synthesizing final answer..."
        )

        final_answer = self._synthesize_answer(
            question=question,
            all_results=all_results,
            is_comparison=is_comparison,
        )

        # =====================================================
        # Step 5: Sources
        # =====================================================

        print(
            "\n📚 Step 5: Extracting sources..."
        )

        sources = self._extract_sources(
            all_results
        )

        print(
            f"   Sources: {len(sources)}"
        )

        print(
            "\n================================================="
        )

        print(
            "✅ Agentic RAG completed"
        )

        print(
            "=================================================\n"
        )

        return {
            "answer": final_answer,
            "sub_queries": sub_queries,
            "search_plans": search_plans,
            "sources": sources,
            "num_searches": len(sub_queries),
            "is_comparison": is_comparison,
        }

    # ============================================================
    # COMPARISON DETECTION
    # ============================================================

    @staticmethod
    def _is_comparison_question(
        question: str,
    ) -> bool:

        query = (
            question
            or ""
        ).lower()

        comparison_keywords = [

            # English

            "which",
            "better",
            "best",
            "compare",
            "comparison",
            "difference",
            "differences",
            "similarity",
            "similarities",
            "more informative",
            "less informative",
            "more useful",
            "less useful",
            "stronger",
            "weaker",

            # Persian

            "کدام",
            "کدوم",
            "بهتر",
            "بهترین",
            "مقایسه",
            "تفاوت",
            "فرق",
            "شباهت",
            "اطلاعات بیشتری",
            "اطلاعات بهتر",
            "کامل‌تر",
            "کامل تر",
            "جامع‌تر",
            "جامع تر",
        ]

        return any(
            keyword in query
            for keyword in comparison_keywords
        )

    # ============================================================
    # COMPARISON SUB-QUERIES
    # ============================================================

    def _build_comparison_sub_queries(
        self,
        question: str,
    ) -> List[str]:

        """
        Build retrieval-oriented queries for comparison.

        We deliberately do NOT ask the LLM to answer here.

        Example:

            Which article provides better information
            about ML?

        becomes approximately:

            1. What information about ML is discussed?
            2. What ML concepts, methods, models and applications
               are discussed?
            3. What is the depth and breadth of ML coverage?
        """

        if not self.ollama_available:
            return [
                question,
                f"Machine learning information discussed in: {question}",
                f"Machine learning methods concepts and applications: {question}",
            ]

        prompt = f"""
You are a retrieval query planner.

The user asks a comparison question.

Create at most 3 search queries that will help retrieve
evidence needed to compare the documents.

Rules:

1. Do NOT answer the question.
2. Do NOT decide which document is better.
3. Focus on retrieving factual evidence.
4. Cover the topic, concepts, methods, details, and breadth
   relevant to the comparison.
5. Return ONLY a JSON array of strings.
6. Use the same language as the user's question.

User question:

{question}

JSON:
"""

        try:

            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                },
            )

            text = self._extract_ollama_response(
                response
            )

            parsed = self._parse_json_array(
                text
            )

            if not parsed:
                return [question]

            cleaned = []

            for item in parsed:

                if not isinstance(
                    item,
                    str,
                ):
                    continue

                item = item.strip()

                if (
                    item
                    and item not in cleaned
                ):
                    cleaned.append(item)

            return cleaned[
                :self.MAX_SUB_QUERIES
            ] or [question]

        except Exception:

            logger.exception(
                "Comparison query decomposition failed"
            )

            return [question]

    # ============================================================
    # NORMAL DECOMPOSITION
    # ============================================================

    def _decompose_question(
        self,
        question: str,
    ) -> List[str]:

        if not self.ollama_available:
            return [question]

        prompt = f"""
You are a query decomposition expert.

Break the user's question into at most 3 independent
retrieval queries.

Rules:

1. If the question is simple, return exactly one query.
2. Every query must help answer the original question.
3. Do not invent information.
4. Do not answer the question.
5. Return ONLY valid JSON.
6. The JSON must be an array of strings.
7. Preserve important technical terms.

User question:

{question}

JSON:
"""

        try:

            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                },
            )

            text = self._extract_ollama_response(
                response
            )

            parsed = self._parse_json_array(
                text
            )

            if not parsed:
                return [question]

            cleaned = []

            for item in parsed:

                if not isinstance(
                    item,
                    str,
                ):
                    continue

                item = item.strip()

                if (
                    item
                    and item not in cleaned
                ):
                    cleaned.append(item)

            return cleaned[
                :self.MAX_SUB_QUERIES
            ] or [question]

        except Exception:

            logger.exception(
                "Question decomposition failed"
            )

            return [question]

    # ============================================================
    # SEARCH PLANNING
    # ============================================================

    def _decide_search_type(
        self,
        sub_query: str,
    ) -> str:

        query = (
            sub_query
            or ""
        ).lower().strip()

        graph_keywords_fa = [
            "رابطه",
            "ارتباط",
            "ارجاع",
            "منبع",
            "نویسنده",
            "مرتبط",
            "وابسته",
            "چه کسی",
        ]

        vector_keywords_fa = [
            "چیست",
            "چیه",
            "تعریف",
            "معنی",
            "توضیح",
            "شرح",
            "چگونه",
            "چطور",
            "چرا",
            "اطلاعات",
            "محتوا",
            "موضوع",
        ]

        graph_keywords_en = [
            "relationship",
            "relation",
            "citation",
            "reference",
            "author",
            "related",
            "connected",
            "dependency",
        ]

        vector_keywords_en = [
            "definition",
            "meaning",
            "what is",
            "explain",
            "describe",
            "how",
            "why",
            "information",
            "content",
            "topic",
            "concept",
            "method",
        ]

        comparison_keywords = [
            "compare",
            "comparison",
            "which",
            "better",
            "best",
            "difference",
            "differences",
            "کدام",
            "بهتر",
            "مقایسه",
            "تفاوت",
            "فرق",
        ]

        graph_match = any(
            keyword in query
            for keyword in (
                graph_keywords_fa
                + graph_keywords_en
            )
        )

        vector_match = any(
            keyword in query
            for keyword in (
                vector_keywords_fa
                + vector_keywords_en
            )
        )

        comparison_match = any(
            keyword in query
            for keyword in comparison_keywords
        )

        if comparison_match:
            return "hybrid"

        if graph_match and vector_match:
            return "hybrid"

        if graph_match:
            return "graph"

        if vector_match:
            return "vector"

        return "hybrid"

    # ============================================================
    # COMPARISON SEARCH
    # ============================================================

    def _comparison_search(
        self,
        question: str,
        sub_queries: List[str],
        search_plans: List[str],
        document_ids: List[int],
        top_k: int,
    ) -> List[Dict[str, Any]]:

        """
        Search each document independently.

        This is the most important part for questions like:

            Which article is better?

        Without this, a document with many high-scoring chunks
        can dominate the entire retrieval context.
        """

        all_results = []

        for document_id in document_ids:

            print(
                f"\n   📄 Comparing Document {document_id}"
            )

            for index, (
                sub_query,
                plan,
            ) in enumerate(
                zip(
                    sub_queries,
                    search_plans,
                ),
                start=1,
            ):

                print(
                    f"      🔎 Query "
                    f"{index}/{len(sub_queries)}"
                )

                print(
                    f"         {sub_query}"
                )

                try:

                    # -----------------------------------------
                    # Vector
                    # -----------------------------------------

                    vector_results = (
                        self._vector_search(
                            query=sub_query,
                            document_ids=[document_id],
                            top_k=top_k,
                        )
                    )

                    # -----------------------------------------
                    # Graph
                    # -----------------------------------------

                    graph_results = []

                    if plan in (
                        "graph",
                        "hybrid",
                    ):

                        graph_results = (
                            self._graph_search(
                                query=sub_query,
                                document_ids=[document_id],
                            )
                        )

                    # -----------------------------------------
                    # Merge
                    # -----------------------------------------

                    if plan == "vector":

                        results = vector_results

                    elif plan == "graph":

                        results = graph_results

                    else:

                        results = self._merge_results(
                            vector_results,
                            graph_results,
                            limit=top_k,
                        )

                    for result in results:

                        result[
                            "retrieval_document_id"
                        ] = document_id

                    print(
                        f"         Results: "
                        f"{len(results)}"
                    )

                    all_results.append(
                        {
                            "sub_query": sub_query,
                            "search_type": plan,
                            "document_id": document_id,
                            "results": results,
                        }
                    )

                except Exception as e:

                    logger.exception(
                        "Comparison search failed"
                    )

                    print(
                        f"         ⚠️ Error: {e}"
                    )

                    all_results.append(
                        {
                            "sub_query": sub_query,
                            "search_type": plan,
                            "document_id": document_id,
                            "results": [],
                        }
                    )

        return all_results

    # ============================================================
    # VECTOR SEARCH
    # ============================================================

    def _vector_search(
        self,
        query: str,
        document_ids: Optional[List[int]],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:

        if not query:
            return []

        if not self.embedding_service:
            print(
                "   ⚠️ EmbeddingService unavailable"
            )
            return []

        if not self.qdrant_client:
            print(
                "   ⚠️ Qdrant unavailable"
            )
            return []

        try:

            query_embedding = (
                self.embedding_service
                .generate_embedding(query)
            )

            query_filter = (
                self._build_document_filter(
                    document_ids
                )
            )

            response = (
                self.qdrant_client
                .query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,
                    query_filter=query_filter,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
            )

            points = getattr(
                response,
                "points",
                [],
            )

            results = []

            for point in points:

                payload = (
                    point.payload
                    or {}
                )

                text = str(
                    payload.get(
                        "text",
                        "",
                    )
                ).strip()

                if not text:
                    continue

                results.append(
                    {
                        "text": text,

                        "score": float(
                            point.score
                            if point.score is not None
                            else 0.0
                        ),

                        "doc_id": payload.get(
                            "doc_id"
                        ),

                        "chunk_id": payload.get(
                            "chunk_id"
                        ),

                        "chunk_index": payload.get(
                            "chunk_index"
                        ),

                        "total_chunks": payload.get(
                            "total_chunks"
                        ),

                        "document_title": payload.get(
                            "document_title",
                            "",
                        ),

                        "document_filename": payload.get(
                            "document_filename",
                            "",
                        ),

                        "document_path": payload.get(
                            "document_path",
                            "",
                        ),

                        "type": "vector",
                    }
                )

            return results

        except Exception as e:

            logger.exception(
                "Qdrant vector search failed"
            )

            print(
                f"   ⚠️ Vector search error: {e}"
            )

            return []

    # ============================================================
    # DOCUMENT FILTER
    # ============================================================

    def _build_document_filter(
        self,
        document_ids: Optional[List[int]],
    ) -> Optional[Filter]:

        if not document_ids:
            return None

        normalized_ids = (
            self._normalize_document_ids(
                document_ids
            )
        )

        if not normalized_ids:
            return None

        return Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(
                        any=normalized_ids
                    ),
                )
            ]
        )

    # ============================================================
    # GRAPH SEARCH
    # ============================================================

    def _graph_search(
        self,
        query: str,
        document_ids: Optional[List[int]],
    ) -> List[Dict[str, Any]]:

        if not query:
            return []

        if not self.neo4j_service:
            return []

        if not self.neo4j_service.driver:
            return []

        try:

            search_terms = (
                self._extract_search_terms(
                    query
                )
            )

            if not search_terms:
                search_terms = [query]

            results = []

            with self.neo4j_service.driver.session() as session:

                if document_ids:

                    cypher = """
                    MATCH (d:Document)-[:CONTAINS]->(e:Entity)

                    WHERE d.id IN $document_ids

                      AND any(
                          term IN $terms
                          WHERE
                            toLower(coalesce(e.name, ''))
                            CONTAINS toLower(term)
                      )

                    RETURN
                        e.name AS entity_name,
                        e.type AS entity_type,
                        e.description AS description,
                        d.id AS doc_id

                    LIMIT 20
                    """

                    records = session.run(
                        cypher,
                        {
                            "document_ids": document_ids,
                            "terms": search_terms,
                        },
                    )

                else:

                    cypher = """
                    MATCH (d:Document)-[:CONTAINS]->(e:Entity)

                    WHERE any(
                        term IN $terms
                        WHERE
                            toLower(coalesce(e.name, ''))
                            CONTAINS toLower(term)
                    )

                    RETURN
                        e.name AS entity_name,
                        e.type AS entity_type,
                        e.description AS description,
                        d.id AS doc_id

                    LIMIT 20
                    """

                    records = session.run(
                        cypher,
                        {
                            "terms": search_terms,
                        },
                    )

                for record in records:

                    entity_name = (
                        record.get(
                            "entity_name"
                        )
                        or ""
                    )

                    entity_type = (
                        record.get(
                            "entity_type"
                        )
                        or "Entity"
                    )

                    description = (
                        record.get(
                            "description"
                        )
                        or ""
                    )

                    doc_id = record.get(
                        "doc_id"
                    )

                    text = (
                        f"Entity: {entity_name}\n"
                        f"Type: {entity_type}"
                    )

                    if description:

                        text += (
                            f"\nDescription: "
                            f"{description}"
                        )

                    results.append(
                        {
                            "text": text,

                            # IMPORTANT:
                            # Do not use 1.0 because that can
                            # dominate vector similarity.
                            "score": 0.5,

                            "doc_id": doc_id,

                            "chunk_id": None,

                            "chunk_index": None,

                            "total_chunks": None,

                            "type": "graph",

                            "entity_name": entity_name,

                            "entity_type": entity_type,

                            "description": description,
                        }
                    )

            return results

        except Exception as e:

            logger.exception(
                "Neo4j graph search failed"
            )

            print(
                f"   ⚠️ Graph search error: {e}"
            )

            return []

    # ============================================================
    # SEARCH TERMS
    # ============================================================

    @staticmethod
    def _extract_search_terms(
        query: str,
    ) -> List[str]:

        query = (
            query
            .strip()
        )

        if not query:
            return []

        terms = re.findall(
            r"[A-Za-z0-9_]+|[\u0600-\u06FF]+",
            query,
        )

        terms = [
            term.strip()
            for term in terms
            if len(term.strip()) >= 2
        ]

        stop_words = {

            # English

            "what",
            "is",
            "are",
            "the",
            "a",
            "an",
            "of",
            "and",
            "or",
            "in",
            "on",
            "to",
            "for",
            "with",
            "how",
            "why",
            "does",
            "do",
            "which",
            "article",
            "articles",
            "better",
            "best",
            "information",

            # Persian

            "چیست",
            "چی",
            "چه",
            "است",
            "هست",
            "این",
            "آن",
            "یک",
            "در",
            "از",
            "به",
            "با",
            "برای",
            "و",
            "یا",
            "را",
            "که",
            "کدام",
            "کدوم",
            "بهتر",
            "بهترین",
            "مقاله",
            "مقالات",
            "اطلاعات",
        }

        filtered = [
            term
            for term in terms
            if term.lower()
            not in stop_words
        ]

        return filtered[:10]

    # ============================================================
    # MERGE RESULTS
    # ============================================================

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        limit: int = DEFAULT_TOP_K,
    ) -> List[Dict[str, Any]]:

        merged = (
            vector_results
            + graph_results
        )

        if not merged:
            return []

        unique_results = []

        seen = set()

        for result in merged:

            result_type = result.get(
                "type",
                "unknown",
            )

            chunk_id = result.get(
                "chunk_id"
            )

            entity_name = result.get(
                "entity_name"
            )

            text = (
                result.get(
                    "text",
                    "",
                )
                or ""
            )

            if chunk_id:

                key = (
                    "chunk",
                    str(chunk_id),
                )

            elif entity_name:

                key = (
                    "entity",
                    result.get("doc_id"),
                    str(
                        entity_name
                    ).lower(),
                )

            else:

                key = (
                    result_type,
                    result.get(
                        "doc_id"
                    ),
                    re.sub(
                        r"\s+",
                        " ",
                        text[:150].lower(),
                    ),
                )

            if key in seen:
                continue

            seen.add(key)

            unique_results.append(
                result
            )

        unique_results.sort(
            key=lambda item: float(
                item.get(
                    "score",
                    0.0,
                )
                or 0.0
            ),
            reverse=True,
        )

        return unique_results[
            :limit
        ]

    # ============================================================
    # SYNTHESIS
    # ============================================================

    def _synthesize_answer(
        self,
        question: str,
        all_results: List[Dict[str, Any]],
        is_comparison: bool = False,
    ) -> str:

        context = self._build_context(
            all_results
        )

        if not context.strip():

            return (
                "اطلاعات کافی برای پاسخ به این سؤال "
                "در اسناد موجود پیدا نشد."
            )

        if not self.ollama_available:

            return (
                "Ollama در دسترس نیست و امکان تولید "
                "پاسخ نهایی وجود ندارد.\n\n"
                "نتایج بازیابی‌شده:\n"
                f"{context}"
            )

        # =====================================================
        # Comparison prompt
        # =====================================================

        if is_comparison:

            prompt = f"""
You are a precise research assistant.

The user is asking a comparison question.

Your job is to compare the documents using ONLY
the retrieved context.

CRITICAL RULES:

1. Do not use outside knowledge.
2. Do not invent facts.
3. Treat each document independently.
4. Do not conclude that information is missing merely
   because one particular chunk does not contain it.
5. Compare the actual retrieved evidence.
6. Identify which document provides broader, deeper,
   clearer, or more useful information for the requested topic.
7. Explain WHY one document is better if the evidence
   supports such a conclusion.
8. If the documents are similar, say so.
9. If the evidence really is insufficient, explicitly say so.
10. Mention document IDs and titles whenever available.
11. Use concrete evidence from the retrieved text.
12. Do not mention internal RAG implementation details.
13. Answer in the same language as the user's question.

IMPORTANT:

The question "which article is better?" is NOT asking
whether the documents are generally good.

It is asking which document provides better information
for the specific topic mentioned by the user.

User question:

{question}

Retrieved context:

{context}

Final answer:
"""

        else:

            prompt = f"""
You are a precise research assistant.

Answer the user's question ONLY using the provided
retrieved context.

Rules:

1. Do not invent facts.
2. Do not use knowledge outside the context.
3. If the context is insufficient, explicitly say so.
4. Prefer direct and concise answers.
5. Mention document ID and chunk number when useful.
6. If different sources conflict, explain the conflict.
7. Do not mention internal RAG implementation details.
8. Answer in the same language as the user's question.

User question:

{question}

Retrieved context:

{context}

Final answer:
"""

        try:

            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                },
            )

            answer = (
                self._extract_ollama_response(
                    response
                )
                .strip()
            )

            if not answer:

                return (
                    "پاسخ معتبری از مدل دریافت نشد."
                )

            return answer

        except Exception as e:

            logger.exception(
                "Answer synthesis failed"
            )

            return (
                "خطا در تولید پاسخ نهایی: "
                f"{e}"
            )

    # ============================================================
    # CONTEXT
    # ============================================================

    def _build_context(
        self,
        all_results: List[Dict[str, Any]],
    ) -> str:

        sections = []

        source_number = 1

        # -----------------------------------------------------
        # Comparison documents should remain grouped.
        # -----------------------------------------------------

        for search_index, item in enumerate(
            all_results,
            start=1,
        ):

            sub_query = item.get(
                "sub_query",
                "",
            )

            search_type = item.get(
                "search_type",
                "unknown",
            )

            document_id = item.get(
                "document_id"
            )

            results = item.get(
                "results",
                [],
            )

            if not results:
                continue

            section_lines = []

            section_lines.append(
                f"SEARCH {search_index}"
            )

            section_lines.append(
                f"Sub-question: {sub_query}"
            )

            section_lines.append(
                f"Search type: {search_type}"
            )

            if document_id is not None:

                section_lines.append(
                    f"Document scope: {document_id}"
                )

            for result in results[
                :self.MAX_CONTEXT_RESULTS_PER_QUERY
            ]:

                doc_id = result.get(
                    "doc_id",
                    document_id or "unknown",
                )

                chunk_index = result.get(
                    "chunk_index"
                )

                result_type = result.get(
                    "type",
                    "unknown",
                )

                score = result.get(
                    "score",
                    0,
                )

                document_title = result.get(
                    "document_title",
                    "",
                )

                document_filename = result.get(
                    "document_filename",
                    "",
                )

                text = (
                    result.get(
                        "text",
                        "",
                    )
                    or ""
                ).strip()

                if not text:
                    continue

                source_label = (
                    f"SOURCE {source_number}"
                )

                section_lines.append(
                    f"\n[{source_label}]"
                )

                section_lines.append(
                    f"document_id={doc_id}"
                )

                if document_title:

                    section_lines.append(
                        f"document_title="
                        f"{document_title}"
                    )

                if document_filename:

                    section_lines.append(
                        f"document_filename="
                        f"{document_filename}"
                    )

                if chunk_index is not None:

                    section_lines.append(
                        f"chunk_index="
                        f"{chunk_index}"
                    )

                section_lines.append(
                    f"type={result_type}"
                )

                section_lines.append(
                    f"score={score}"
                )

                # Limit individual context size.

                text = text[
                    :self.MAX_CONTEXT_CHARS_PER_RESULT
                ]

                section_lines.append(
                    f"text={text}"
                )

                source_number += 1

            sections.append(
                "\n".join(
                    section_lines
                )
            )

        return "\n\n".join(
            sections
        )

    # ============================================================
    # SOURCE EXTRACTION
    # ============================================================

    def _extract_sources(
        self,
        all_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        sources = []

        seen = set()

        for item in all_results:

            sub_query = item.get(
                "sub_query",
                "",
            )

            search_type = item.get(
                "search_type",
                "unknown",
            )

            for result in item.get(
                "results",
                [],
            ):

                doc_id = result.get(
                    "doc_id"
                )

                if doc_id is None:
                    continue

                chunk_id = result.get(
                    "chunk_id"
                )

                chunk_index = result.get(
                    "chunk_index"
                )

                entity_name = result.get(
                    "entity_name"
                )

                if chunk_id:

                    key = (
                        "chunk",
                        doc_id,
                        chunk_id,
                    )

                elif entity_name:

                    key = (
                        "entity",
                        doc_id,
                        entity_name.lower(),
                    )

                else:

                    key = (
                        "document",
                        doc_id,
                    )

                if key in seen:
                    continue

                seen.add(key)

                source = {
                    "document_id": doc_id,

                    "document_title": (
                        result.get(
                            "document_title",
                            "",
                        )
                    ),

                    "document_filename": (
                        result.get(
                            "document_filename",
                            "",
                        )
                    ),

                    "search_type": search_type,

                    "chunk_id": chunk_id,

                    "chunk_index": chunk_index,

                    "score": result.get(
                        "score",
                        0.0,
                    ),

                    "text": (
                        result.get(
                            "text",
                            "",
                        )
                        or ""
                    )[:300],

                    "sub_query": sub_query,
                }

                if entity_name:

                    source[
                        "entity_name"
                    ] = entity_name

                sources.append(
                    source
                )

        sources.sort(
            key=lambda source: float(
                source.get(
                    "score",
                    0.0,
                )
                or 0.0
            ),
            reverse=True,
        )

        return sources

    # ============================================================
    # QDRANT DEBUG
    # ============================================================

    def debug_documents(
        self,
    ) -> Dict[Any, List[Dict[str, Any]]]:

        if not self.qdrant_client:

            print(
                "\n❌ Qdrant client is unavailable.\n"
            )

            return {}

        try:

            offset = None

            total = 0

            docs = {}

            while True:

                points, offset = (
                    self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                )

                if not points:
                    break

                total += len(points)

                for point in points:

                    payload = (
                        point.payload
                        or {}
                    )

                    doc_id = payload.get(
                        "doc_id"
                    )

                    if doc_id not in docs:

                        docs[doc_id] = []

                    docs[doc_id].append(
                        {
                            "point_id": point.id,

                            "chunk_id": payload.get(
                                "chunk_id"
                            ),

                            "chunk_index": payload.get(
                                "chunk_index"
                            ),

                            "total_chunks": payload.get(
                                "total_chunks"
                            ),

                            "document_title": payload.get(
                                "document_title",
                                "",
                            ),

                            "document_filename": payload.get(
                                "document_filename",
                                "",
                            ),

                            "text": payload.get(
                                "text",
                                "",
                            ),
                        }
                    )

                if offset is None:
                    break

            print(
                "\n================ QDRANT DEBUG ================"
            )

            print(
                f"Collection: {self.collection_name}"
            )

            print(
                f"Total points: {total}"
            )

            print(
                f"Documents: {len(docs)}"
            )

            for doc_id, chunks in docs.items():

                print(
                    f"\n📄 Document ID: {doc_id}"
                )

                print(
                    f"   Chunks: {len(chunks)}"
                )

                chunks.sort(
                    key=lambda chunk: (
                        chunk.get(
                            "chunk_index"
                        )
                        if chunk.get(
                            "chunk_index"
                        ) is not None
                        else 999999
                    )
                )

                for chunk in chunks[:5]:

                    print(
                        "   "
                        f"chunk_index="
                        f"{chunk.get('chunk_index')} "
                        f"chunk_id="
                        f"{chunk.get('chunk_id')}"
                    )

                    if chunk.get(
                        "document_title"
                    ):

                        print(
                            "   "
                            f"title="
                            f"{chunk.get('document_title')}"
                        )

                    text = (
                        chunk.get(
                            "text",
                            "",
                        )
                        or ""
                    )

                    text = re.sub(
                        r"\s+",
                        " ",
                        text,
                    )

                    print(
                        f"   text={text[:200]}"
                    )

            print(
                "\n===============================================\n"
            )

            return docs

        except Exception as e:

            logger.exception(
                "Qdrant debug failed"
            )

            print(
                f"❌ Qdrant debug error: {e}"
            )

            return {}

    # ============================================================
    # COLLECTION DEBUG
    # ============================================================

    def debug_collection(
        self,
    ) -> Dict[str, Any]:

        if not self.qdrant_client:
            return {}

        try:

            info = (
                self.qdrant_client
                .get_collection(
                    self.collection_name
                )
            )

            vectors_config = (
                info.config.params.vectors
            )

            vector_size = getattr(
                vectors_config,
                "size",
                None,
            )

            return {
                "name": self.collection_name,

                "points_count": (
                    info.points_count
                ),

                "vector_size": vector_size,
            }

        except Exception:

            logger.exception(
                "Failed to inspect Qdrant collection"
            )

            return {}

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(
        self,
    ) -> Dict[str, Any]:

        qdrant_ok = False

        neo4j_ok = False

        ollama_ok = False

        embedding_ok = (
            self.embedding_service is not None
        )

        # -----------------------------------------------------
        # Qdrant
        # -----------------------------------------------------

        if self.qdrant_client:

            try:

                self.qdrant_client.get_collections()

                qdrant_ok = True

            except Exception:

                qdrant_ok = False

        # -----------------------------------------------------
        # Neo4j
        # -----------------------------------------------------

        if (
            self.neo4j_service
            and self.neo4j_service.driver
        ):

            try:

                self.neo4j_service.driver.verify_connectivity()

                neo4j_ok = True

            except Exception:

                neo4j_ok = False

        # -----------------------------------------------------
        # Ollama
        # -----------------------------------------------------

        try:

            ollama.list()

            ollama_ok = True

        except Exception:

            ollama_ok = False

        return {
            "qdrant": qdrant_ok,

            "embedding": embedding_ok,

            "neo4j": neo4j_ok,

            "ollama": ollama_ok,

            "model": self.model_name,

            "collection": self.collection_name,
        }

    # ============================================================
    # HELPERS
    # ============================================================

    @staticmethod
    def _normalize_document_ids(
        document_ids: Optional[List[int]],
    ) -> Optional[List[int]]:

        if not document_ids:
            return None

        normalized = []

        for value in document_ids:

            try:

                value = int(value)

            except (
                TypeError,
                ValueError,
            ):

                continue

            if value not in normalized:

                normalized.append(
                    value
                )

        return normalized or None

    @staticmethod
    def _extract_ollama_response(
        response: Any,
    ) -> str:

        if response is None:
            return ""

        if isinstance(
            response,
            dict,
        ):

            return str(
                response.get(
                    "response",
                    "",
                )
            )

        value = getattr(
            response,
            "response",
            None,
        )

        if value is not None:
            return str(value)

        return str(
            response
        )

    @staticmethod
    def _parse_json_array(
        text: str,
    ) -> Optional[List[Any]]:

        if not text:
            return None

        text = text.strip()

        # Remove markdown code fences.

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

        # First attempt.

        try:

            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                list,
            ):

                return parsed

        except json.JSONDecodeError:
            pass

        # Locate JSON array.

        start = text.find("[")

        end = text.rfind("]")

        if (
            start == -1
            or end == -1
            or end <= start
        ):

            return None

        candidate = text[
            start:end + 1
        ]

        try:

            parsed = json.loads(
                candidate
            )

            if isinstance(
                parsed,
                list,
            ):

                return parsed

        except json.JSONDecodeError:

            return None

        return None

    # ============================================================
    # CLOSE
    # ============================================================

    def close(
        self,
    ):

        try:

            if self.neo4j_service:

                self.neo4j_service.close()

        except Exception:

            logger.exception(
                "Error closing Neo4j service"
            )

        try:

            if self.qdrant_client:

                self.qdrant_client.close()

        except Exception:

            logger.exception(
                "Error closing Qdrant client"
            )

        self.qdrant_client = None