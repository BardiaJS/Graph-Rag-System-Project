"""
Neo4j layer.

Schema
    (Document {id, title, user_id})
        -[:HAS_SECTION]->  (Section {key, heading, level, document_id})   top-level sections
    (Section) -[:HAS_SUBSECTION]-> (Section)
    (Section | Document) -[:HAS_CHUNK]-> (Chunk {key, document_id, chunk_index, text, page, headings})
    (Chunk) -[:NEXT]-> (Chunk)                      reading order
    (Chunk) -[:MENTIONS]-> (Entity {key, name, type})          written by entity_service
    (Entity) -[:RELATED_TO {type, document_id, chunk_key}]-> (Entity)
    (Document) -[:HAS_FIGURE]-> (Figure)            written by figure_service
    (Chunk) -[:DESCRIBES|REFERENCES|NEAR]-> (Figure)

Sync with Qdrant
    Chunk.key == f"{document_id}:{chunk_index}", the same pair stored in the Qdrant payload.

Note: Entity nodes are shared across documents (that is what links documents together).
Every query that crosses documents MUST filter chunks by the caller's allowed document ids.
"""
import logging
import os
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

_driver = None
_schema_ready = False


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


def close_driver():
    global _driver, _schema_ready
    if _driver:
        _driver.close()
        _driver = None
        _schema_ready = False


def chunk_key(document_id: int, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index}"


def ensure_schema():
    global _schema_ready
    if _schema_ready:
        return
    statements = [
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT section_key IF NOT EXISTS FOR (s:Section) REQUIRE s.key IS UNIQUE",
        "CREATE CONSTRAINT chunk_key IF NOT EXISTS FOR (c:Chunk) REQUIRE c.key IS UNIQUE",
        "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
        "CREATE CONSTRAINT figure_key IF NOT EXISTS FOR (f:Figure) REQUIRE f.key IS UNIQUE",
        "CREATE INDEX chunk_document IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
    ]
    with get_driver().session() as session:
        for stmt in statements:
            session.run(stmt)
    _schema_ready = True


# ------------------------------------------------------------------ build

def _plan_graph_rows(document_id: int, chunks: List[dict]) -> Dict[str, List[dict]]:
    """Pure function: turn chunks into row batches. Easy to unit-test."""
    sections: Dict[str, dict] = {}
    chunk_rows: List[dict] = []
    next_rows: List[dict] = []

    for idx, chunk in enumerate(chunks):
        path = [h for h in (chunk.get("headings") or []) if h]

        section_key: Optional[str] = None
        for level in range(len(path)):
            key = f"{document_id}::" + " > ".join(path[: level + 1])
            parent = (
                f"{document_id}::" + " > ".join(path[:level]) if level > 0 else None
            )
            sections.setdefault(
                key,
                {"key": key, "heading": path[level], "level": level, "parent": parent},
            )
            section_key = key  # deepest section wins

        chunk_rows.append({
            "key": chunk_key(document_id, idx),
            "idx": idx,
            "text": chunk.get("text", ""),
            "page": chunk.get("page"),
            "headings": path,
            "section_key": section_key,
        })
        if idx > 0:
            next_rows.append({
                "a": chunk_key(document_id, idx - 1),
                "b": chunk_key(document_id, idx),
            })

    section_rows = list(sections.values())
    return {
        "sections": section_rows,
        "sub_edges": [s for s in section_rows if s["parent"]],
        "top_edges": [s for s in section_rows if not s["parent"]],
        "chunks": chunk_rows,
        "in_section": [c for c in chunk_rows if c["section_key"]],
        "no_section": [c for c in chunk_rows if not c["section_key"]],
        "next": next_rows,
    }


def delete_document_graph(document_id: int):
    """Remove everything belonging to a document (safe to call before a rebuild)."""
    ensure_schema()
    with get_driver().session() as session:
        session.run(
            "MATCH ()-[r:RELATED_TO {document_id: $id}]-() DELETE r", id=document_id
        )
        session.run(
            "MATCH (n) WHERE n.document_id = $id DETACH DELETE n", id=document_id
        )
        session.run("MATCH (d:Document {id: $id}) DETACH DELETE d", id=document_id)
        # entities no longer mentioned by any chunk
        session.run("MATCH (e:Entity) WHERE NOT (e)<-[:MENTIONS]-() DETACH DELETE e")


def build_graph(
    document_id: int,
    chunks: List[dict],
    title: str = "",
    user_id: Optional[str] = None,
):
    """Idempotent: deletes any previous graph for this document, then rebuilds in
    one transaction using batched UNWIND queries."""
    document_id = int(document_id)
    delete_document_graph(document_id)
    rows = _plan_graph_rows(document_id, chunks)

    def tx(t):
        t.run(
            "MERGE (d:Document {id: $id}) SET d.title = $title, d.user_id = $user_id",
            id=document_id,
            title=title or f"Document {document_id}",
            user_id=user_id,
        )
        t.run(
            """
            UNWIND $rows AS row
            MERGE (s:Section {key: row.key})
            SET s.heading = row.heading, s.level = row.level, s.document_id = $doc_id
            """,
            rows=rows["sections"], doc_id=document_id,
        )
        t.run(
            """
            UNWIND $rows AS row
            MATCH (p:Section {key: row.parent}) MATCH (s:Section {key: row.key})
            MERGE (p)-[:HAS_SUBSECTION]->(s)
            """,
            rows=rows["sub_edges"],
        )
        t.run(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: $doc_id}) MATCH (s:Section {key: row.key})
            MERGE (d)-[:HAS_SECTION]->(s)
            """,
            rows=rows["top_edges"], doc_id=document_id,
        )
        t.run(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {key: row.key})
            SET c.document_id = $doc_id, c.chunk_index = row.idx, c.text = row.text,
                c.page = row.page, c.headings = row.headings
            """,
            rows=rows["chunks"], doc_id=document_id,
        )
        t.run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk {key: row.key}) MATCH (s:Section {key: row.section_key})
            MERGE (s)-[:HAS_CHUNK]->(c)
            """,
            rows=rows["in_section"],
        )
        t.run(
            """
            UNWIND $rows AS row
            MATCH (d:Document {id: $doc_id}) MATCH (c:Chunk {key: row.key})
            MERGE (d)-[:HAS_CHUNK]->(c)
            """,
            rows=rows["no_section"], doc_id=document_id,
        )
        t.run(
            """
            UNWIND $rows AS row
            MATCH (a:Chunk {key: row.a}) MATCH (b:Chunk {key: row.b})
            MERGE (a)-[:NEXT]->(b)
            """,
            rows=rows["next"],
        )

    with get_driver().session() as session:
        session.execute_write(tx)

    logger.info(
        f"Graph built for document {document_id}: "
        f"{len(rows['chunks'])} chunks, {len(rows['sections'])} sections"
    )
    return True


# ------------------------------------------------------------------ reads

def get_graph_stats(document_id: Optional[int] = None) -> Dict[str, Any]:
    with get_driver().session() as session:
        if document_id is not None:
            result = session.run(
                """
                MATCH (c:Chunk {document_id: $id})
                OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
                RETURN count(DISTINCT c) AS chunks, count(DISTINCT e) AS entities
                """,
                id=document_id,
            )
        else:
            result = session.run("MATCH (n) RETURN count(n) AS total")
        return dict(result.single())


def get_chunks_with_context(
    document_id: int, chunk_index: int, context_size: int = 2
) -> List[dict]:
    """The chunk plus `context_size` chunks before and after it, in reading order.
    chunk_index is contiguous per document, so a range query is exact and cheap."""
    context_size = max(0, int(context_size))
    with get_driver().session() as session:
        return session.run(
            """
            MATCH (c:Chunk {document_id: $doc_id})
            WHERE c.chunk_index >= $lo AND c.chunk_index <= $hi
            RETURN c.chunk_index AS idx, c.text AS text, c.page AS page,
                   c.headings AS headings
            ORDER BY c.chunk_index
            """,
            doc_id=document_id,
            lo=chunk_index - context_size,
            hi=chunk_index + context_size,
        ).data()


def get_section_chunks(document_id: int, heading: str) -> List[dict]:
    """All chunks of a section, including its subsections."""
    with get_driver().session() as session:
        return session.run(
            """
            MATCH (s:Section {heading: $heading, document_id: $doc_id})
                  -[:HAS_SUBSECTION*0..]->(:Section)-[:HAS_CHUNK]->(c:Chunk)
            RETURN DISTINCT c.chunk_index AS idx, c.text AS text, c.page AS page,
                   c.headings AS headings
            ORDER BY idx
            """,
            heading=heading,
            doc_id=document_id,
        ).data()


def get_related_sections(document_id: int, heading: str) -> List[dict]:
    """Sibling sections (same parent)."""
    with get_driver().session() as session:
        return session.run(
            """
            MATCH (s:Section {heading: $heading, document_id: $doc_id})
            OPTIONAL MATCH (p:Section)-[:HAS_SUBSECTION]->(s)
            MATCH (sib:Section {document_id: $doc_id})
            WHERE sib <> s AND (
                  (p IS NOT NULL AND (p)-[:HAS_SUBSECTION]->(sib))
               OR (p IS NULL AND sib.level = s.level AND NOT ()-[:HAS_SUBSECTION]->(sib)))
            RETURN DISTINCT sib.heading AS heading, sib.level AS level
            ORDER BY heading
            LIMIT 10
            """,
            heading=heading,
            doc_id=document_id,
        ).data()


def find_chunks_via_entities(
    chunk_keys: List[str],
    allowed_document_ids: List[int],
    limit: int = 6,
) -> List[dict]:
    """Chunks (in any ALLOWED document) that share entities with the given chunks.
    This is the cross-document link: two papers mentioning the same dataset or metric."""
    if not chunk_keys or not allowed_document_ids:
        return []
    with get_driver().session() as session:
        return session.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(o:Chunk)
            WHERE c.key IN $keys
              AND NOT o.key IN $keys
              AND o.document_id IN $docs
            RETURN o.document_id AS document_id, o.chunk_index AS idx, o.text AS text,
                   o.page AS page, o.headings AS headings,
                   count(DISTINCT e) AS shared,
                   collect(DISTINCT e.name)[..5] AS via
            ORDER BY shared DESC
            LIMIT $limit
            """,
            keys=chunk_keys,
            docs=[int(d) for d in allowed_document_ids],
            limit=limit,
        ).data()


def get_figures(document_id: int) -> List[dict]:
    """All figures of a document, in document order."""
    with get_driver().session() as session:
        return session.run(
            """
            MATCH (:Document {id: $id})-[:HAS_FIGURE]->(f:Figure)
            RETURN f.figure_index AS figure_index, f.label AS label, f.number AS number,
                   f.caption AS caption, f.page AS page, f.width AS width,
                   f.height AS height
            ORDER BY f.page, f.figure_index
            """,
            id=int(document_id),
        ).data()


def get_figures_for_chunks(chunk_keys: List[str]) -> List[dict]:
    """Figures linked to the given chunks (caption chunk, in-text mention, or nearby)."""
    if not chunk_keys:
        return []
    with get_driver().session() as session:
        return session.run(
            """
            MATCH (c:Chunk)-[r:DESCRIBES|REFERENCES|NEAR]->(f:Figure)
            WHERE c.key IN $keys
            RETURN c.key AS chunk_key, type(r) AS rel, f.document_id AS document_id,
                   f.figure_index AS figure_index, f.label AS label, f.number AS number,
                   f.caption AS caption, f.page AS page
            """,
            keys=chunk_keys,
        ).data()


_DOC_STATS = """
    CALL { WITH d MATCH (c:Chunk {document_id: d.id}) RETURN count(c) AS chunks }
    CALL { WITH d OPTIONAL MATCH (d)-[:HAS_FIGURE]->(f:Figure) RETURN count(f) AS figures }
    CALL { WITH d OPTIONAL MATCH (:Chunk {document_id: d.id})-[:MENTIONS]->(e:Entity)
           RETURN count(DISTINCT e) AS entities }
    RETURN d.id AS document_id, d.title AS title, d.user_id AS user_id,
           chunks, figures, entities
"""


def list_documents(user_id: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[dict]:
    """Documents in the graph with their sizes. user_id=None lists everything
    (internal/admin use); Laravel should always pass the owner's id."""
    with get_driver().session() as session:
        return session.run(
            "MATCH (d:Document) WHERE $user_id IS NULL OR d.user_id = $user_id "
            + _DOC_STATS
            + " ORDER BY d.id SKIP $skip LIMIT $limit",
            user_id=None if user_id is None else str(user_id),
            skip=int(offset),
            limit=int(limit),
        ).data()


def get_document(document_id: int) -> Optional[dict]:
    with get_driver().session() as session:
        rows = session.run(
            "MATCH (d:Document {id: $id}) " + _DOC_STATS,
            id=int(document_id),
        ).data()
    return rows[0] if rows else None
