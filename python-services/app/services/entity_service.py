"""
Entity and relation extraction -> Neo4j.

One LLM call per chunk extracts entities (methods, datasets, metrics, concepts, authors)
and relations between them. Entities are keyed by (type, normalized name) and shared
across documents, so a metric mentioned in two papers becomes ONE node that links both.
"""
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services import llm_service
from app.services.graph_service import chunk_key, ensure_schema, get_driver

logger = logging.getLogger(__name__)

ENTITY_TYPES = {"METHOD", "DATASET", "METRIC", "CONCEPT", "AUTHOR"}
RELATION_TYPES = {"USES", "EXTENDS", "COMPARES_TO", "EVALUATES_ON", "MEASURED_BY"}

MIN_CHUNK_CHARS = int(os.getenv("ENTITY_MIN_CHUNK_CHARS", 150))
MAX_CHUNK_CHARS = 2000
MAX_ENTITIES_PER_CHUNK = 8
# Stop (and report) after this many model failures in a row instead of grinding through every chunk.
MAX_CONSECUTIVE_LLM_FAILURES = int(os.getenv("ENTITY_MAX_LLM_FAILURES", 3))
MAX_RELATIONS_PER_CHUNK = 6


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _entity_key(etype: str, name: str) -> str:
    return f"{etype}:{normalize(name)}"


PROMPT = """Extract a small knowledge graph from this passage of an academic paper.

Entity types: METHOD, DATASET, METRIC, CONCEPT, AUTHOR.
Relation types: USES, EXTENDS, COMPARES_TO, EVALUATES_ON, MEASURED_BY.

Rules:
- Only include entities explicitly named in the passage. Do not infer.
- Use the shortest canonical name (e.g. "BERT", not "the BERT model").
- Relations may only connect entities you listed. At most {max_e} entities and {max_r} relations.
- If nothing qualifies, return empty lists.

Return JSON only:
{{"entities": [{{"name": "...", "type": "..."}}],
  "relations": [{{"source": "...", "target": "...", "type": "..."}}]}}

Passage:
{text}"""


def _parse(raw: Dict[str, Any], text: str) -> Tuple[List[dict], List[dict]]:
    """Validate LLM output: whitelist types, drop names not literally in the passage
    (cheap hallucination guard), dedupe, cap sizes."""
    haystack = text.lower()
    entities: Dict[str, dict] = {}

    for item in raw.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip().upper()
        if etype not in ENTITY_TYPES or not (2 <= len(name) <= 80):
            continue
        if normalize(name) not in haystack:
            continue
        key = _entity_key(etype, name)
        entities.setdefault(key, {"key": key, "name": name, "type": etype})
        if len(entities) >= MAX_ENTITIES_PER_CHUNK:
            break

    by_name = {normalize(e["name"]): e["key"] for e in entities.values()}
    relations: List[dict] = []
    for item in raw.get("relations") or []:
        if not isinstance(item, dict):
            continue
        rtype = str(item.get("type", "")).strip().upper()
        src = by_name.get(normalize(str(item.get("source", ""))))
        dst = by_name.get(normalize(str(item.get("target", ""))))
        if rtype in RELATION_TYPES and src and dst and src != dst:
            relations.append({"source": src, "target": dst, "type": rtype})
        if len(relations) >= MAX_RELATIONS_PER_CHUNK:
            break

    return list(entities.values()), relations


def extract_from_chunk(text: str) -> Tuple[List[dict], List[dict]]:
    text = (text or "")[:MAX_CHUNK_CHARS]
    if len(text) < MIN_CHUNK_CHARS:
        return [], []
    raw = llm_service.chat_json(
        [{"role": "user", "content": PROMPT.format(
            text=text, max_e=MAX_ENTITIES_PER_CHUNK, max_r=MAX_RELATIONS_PER_CHUNK)}],
        default={},
        strict=True,  # a dead model must raise, not look like "no entities found"
    )
    return _parse(raw, text)


def extract_and_store_entities(
    document_id: int,
    chunks: List[dict],
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, int]:
    """Extract from every chunk, then write everything in two batched queries.
    One chunk failing never aborts the document."""
    document_id = int(document_id)
    mention_rows: List[dict] = []
    relation_rows: List[dict] = []

    consecutive_failures = 0
    for idx, chunk in enumerate(chunks):
        if on_progress and idx % 5 == 0:
            on_progress(idx, len(chunks))
        if len((chunk.get("text") or "")[:MAX_CHUNK_CHARS]) < MIN_CHUNK_CHARS:
            continue  # too short to hold entities; no model call, so it can't count as a failure
        try:
            entities, relations = extract_from_chunk(chunk.get("text", ""))
            consecutive_failures = 0
        except llm_service.LLMError as e:
            consecutive_failures += 1
            logger.warning(f"Entity extraction: model call failed for chunk {idx}: {e}")
            if consecutive_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                raise llm_service.LLMError(
                    f"Language model unavailable: entity extraction stopped after "
                    f"{consecutive_failures} consecutive failures ({e})"
                ) from e
            continue
        except Exception as e:
            logger.warning(f"Entity extraction failed for chunk {idx}: {e}")
            continue
        ck = chunk_key(document_id, idx)
        for e in entities:
            mention_rows.append({"chunk_key": ck, **e})
        for r in relations:
            relation_rows.append({"chunk_key": ck, "document_id": document_id, **r})

        if (idx + 1) % 20 == 0:
            logger.info(f"Entity extraction: {idx + 1}/{len(chunks)} chunks")

    ensure_schema()
    with get_driver().session() as session:
        if mention_rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {key: row.chunk_key})
                MERGE (e:Entity {key: row.key})
                  ON CREATE SET e.name = row.name, e.type = row.type
                MERGE (c)-[:MENTIONS]->(e)
                """,
                rows=mention_rows,
            )
        if relation_rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Entity {key: row.source}) MATCH (b:Entity {key: row.target})
                MERGE (a)-[r:RELATED_TO {type: row.type, document_id: row.document_id}]->(b)
                SET r.chunk_key = row.chunk_key
                """,
                rows=relation_rows,
            )

    stats = {
        "mentions": len(mention_rows),
        "relations": len(relation_rows),
        "entities": len({m["key"] for m in mention_rows}),
    }
    logger.info(f"Entities stored for document {document_id}: {stats}")
    return stats
