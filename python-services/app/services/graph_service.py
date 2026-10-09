import os
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def close_driver():
    global _driver
    if _driver:
        _driver.close()
        _driver = None


def build_graph(document_id: int, chunks: list, title: str = ""):
    """از chunkها گراف بساز"""
    driver = get_driver()

    with driver.session() as session:
        # ۱. Document
        session.run(
            "MERGE (d:Document {id: $id}) SET d.title = $title",
            id=document_id,
            title=title or f"Document {document_id}",
        )

        # ۲. Sectionها (از headings)
        for chunk in chunks:
            headings = chunk.get("headings", [])
            if not headings:
                continue

            for level, heading in enumerate(headings):
                if not heading:
                    continue
                session.run(
                    """
                    MERGE (s:Section {heading: $heading, document_id: $doc_id})
                    SET s.level = $level
                    """,
                    heading=heading,
                    doc_id=document_id,
                    level=level,
                )

        # ۳. Chunkها
        for idx, chunk in enumerate(chunks):
            headings = chunk.get("headings", [])
            primary_heading = headings[-1] if headings else None

            session.run(
                """
                MERGE (c:Chunk {document_id: $doc_id, chunk_index: $idx})
                SET c.text = $text,
                    c.page = $page,
                    c.enriched_text = $enriched
                """,
                doc_id=document_id,
                idx=idx,
                text=chunk.get("text", ""),
                page=chunk.get("page"),
                enriched=chunk.get("enriched_text", ""),
            )

            # Section → Chunk
            if primary_heading:
                session.run(
                    """
                    MATCH (s:Section {heading: $heading, document_id: $doc_id})
                    MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
                    MERGE (s)-[:HAS_CHUNK]->(c)
                    """,
                    heading=primary_heading,
                    doc_id=document_id,
                    idx=idx,
                )

        # ۴. Document → Section
        session.run(
            """
            MATCH (d:Document {id: $doc_id})
            MATCH (s:Section {document_id: $doc_id})
            MERGE (d)-[:HAS_SECTION]->(s)
            """,
            doc_id=document_id,
        )

        # ۵. Chunk → NEXT (ترتیب)
        session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})
            WITH c ORDER BY c.chunk_index
            WITH collect(c) AS chunks
            UNWIND range(0, size(chunks)-2) AS i
            WITH chunks[i] AS current, chunks[i+1] AS next
            MERGE (current)-[:NEXT]->(next)
            """,
            doc_id=document_id,
        )

    logger.info(f"Graph built for document {document_id}")
    return True


def get_graph_stats(document_id: int = None):
    """آمار گراف"""
    driver = get_driver()
    with driver.session() as session:
        if document_id:
            result = session.run(
                "MATCH (c:Chunk {document_id: $doc_id}) RETURN count(c) AS chunks",
                doc_id=document_id,
            )
        else:
            result = session.run("MATCH (n) RETURN count(n) AS total")
        return dict(result.single())



def get_chunks_with_context(
    document_id: int,
    chunk_index: int,
    context_size: int = 2,
):
    """
    یه chunk و همسایه‌هاش (NEXT و PREV) رو برگردون.
    
    Args:
        document_id: شناسه داکیومنت
        chunk_index: ایندکس chunk مرکزی
        context_size: تعداد chunk قبل و بعد
    
    Returns:
        لیست chunkها با ترتیب
    """
    driver = get_driver()
    
    with driver.session() as session:
        # chunkهای قبل
        before = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
            MATCH (c)<-[:NEXT*1..2]-(prev:Chunk)
            RETURN prev.chunk_index AS idx, prev.text AS text, prev.page AS page
            ORDER BY prev.chunk_index ASC
            """,
            """
            MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
            MATCH (c)<-[:NEXT*1..{context_size}]-(prev:Chunk)
            RETURN prev.chunk_index AS idx, prev.text AS text, prev.page AS page
            ORDER BY prev.chunk_index ASC
            """.replace("{context_size}", str(context_size)),
            doc_id=document_id,
            idx=chunk_index,
        ).data()
        
        # chunk مرکزی
        center = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
            RETURN c.chunk_index AS idx, c.text AS text, c.page AS page
            """,
            doc_id=document_id,
            idx=chunk_index,
        ).data()
        
        # chunkهای بعد
        after = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
            MATCH (c)-[:NEXT*1..{context_size}]->(next:Chunk)
            RETURN next.chunk_index AS idx, next.text AS text, next.page AS page
            ORDER BY next.chunk_index ASC
            """.replace("{context_size}", str(context_size)),
            doc_id=document_id,
            idx=chunk_index,
        ).data()
        
        return before + center + after


def get_section_chunks(document_id: int, heading: str):
    """
    همه chunkهای یه section رو برگردون.
    """
    driver = get_driver()
    
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Section {heading: $heading, document_id: $doc_id})
                  -[:HAS_CHUNK]->(c:Chunk)
            RETURN c.chunk_index AS idx, c.text AS text, c.page AS page
            ORDER BY c.chunk_index ASC
            """,
            heading=heading,
            doc_id=document_id,
        )
        return result.data()


def get_related_sections(document_id: int, heading: str):
    """
    sectionهای مرتبط (خواهر/برادر) رو برگردون.
    """
    driver = get_driver()
    
    with driver.session() as session:
        # همون document، sectionهای دیگه
        result = session.run(
            """
            MATCH (s:Section {document_id: $doc_id})
            WHERE s.heading <> $heading
            RETURN s.heading AS heading, s.level AS level
            ORDER BY s.level, s.heading
            LIMIT 10
            """,
            doc_id=document_id,
            heading=heading,
        )
        return result.data()


def get_chunks_with_context(document_id: int, chunk_index: int, context_size: int = 2):
    """chunk + همسایه‌هاش"""
    driver = get_driver()
    with driver.session() as session:
        # چک کن chunk وجود داره
        result = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
            MATCH (c)<-[:NEXT*1..2]-(prev:Chunk)
            RETURN prev.chunk_index AS idx, prev.text AS text, prev.page AS page
            ORDER BY prev.chunk_index ASC
            """,
            doc_id=document_id,
            idx=chunk_index,
        ).data()
        
        center = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
            RETURN c.chunk_index AS idx, c.text AS text, c.page AS page
            """,
            doc_id=document_id,
            idx=chunk_index,
        ).data()
        
        after = session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id, chunk_index: $idx})
            MATCH (c)-[:NEXT*1..2]->(next:Chunk)
            RETURN next.chunk_index AS idx, next.text AS text, next.page AS page
            ORDER BY next.chunk_index ASC
            """,
            doc_id=document_id,
            idx=chunk_index,
        ).data()
        
        return result + center + after


def get_section_chunks(document_id: int, heading: str):
    """همه chunkهای یه section"""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Section {heading: $heading, document_id: $doc_id})
                  -[:HAS_CHUNK]->(c:Chunk)
            RETURN c.chunk_index AS idx, c.text AS text, c.page AS page
            ORDER BY c.chunk_index ASC
            """,
            heading=heading,
            doc_id=document_id,
        )
        return result.data()