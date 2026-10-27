"""
Retrieval primitives. No LLM anywhere in this module.

    collect          vector search (+ neighbour chunks) per query per document, then a graph
                     hop: chunks sharing entities with the direct hits, in ANY allowed document
    select_evidence  top-K per document, in reading order
    attach_figures   figures linked to the chosen chunks
    retrieve         all of the above as one stateless call (backs POST /retrieve)

Scope rule: every function takes the caller's allowed document ids and never searches
outside them, including the entity hop.
"""
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HITS_PER_QUERY = int(os.getenv("RETRIEVAL_HITS_PER_QUERY", os.getenv("AGENT_HITS_PER_QUERY", 6)))
EVIDENCE_PER_DOC = int(os.getenv("RETRIEVAL_EVIDENCE_PER_DOC", os.getenv("AGENT_EVIDENCE_PER_DOC", 6)))
MAX_QUERIES_PER_CALL = int(os.getenv("RETRIEVAL_MAX_QUERIES", 6))
# Reference lists look "similar" to almost any question (names, titles, years) and crowd
# out real content. Their score is multiplied by this unless the user asks about references.
BIBLIOGRAPHY_PENALTY = float(os.getenv("RETRIEVAL_BIBLIOGRAPHY_PENALTY", 0.5))
# Drop demoted chunks entirely once a document has at least this many real ones.
MIN_REAL_EVIDENCE = int(os.getenv("RETRIEVAL_MIN_REAL_EVIDENCE", 3))

_BIB_HEADINGS = ("reference", "bibliograph", "works cited", "منابع", "مراجع", "کتابنامه")
_BIB_LINE = re.compile(r"^\s*(?:[-*\u2022]\s*)?\[\d{1,3}\]\s", re.MULTILINE)
_ASKS_ABOUT_REFS = re.compile(r"referenc|citation|cited|bibliograph|منبع|منابع|مرجع|مراجع|ارجاع", re.IGNORECASE)


def is_bibliography(headings: Optional[List[str]], text: Optional[str]) -> bool:
    """True for a reference-list chunk. Checks the section title and, because neighbour
    chunks carry no headings, also the text: most lines look like "[12] Author, ..."."""
    last = (headings[-1] if headings else "") or ""
    last = re.sub(r"^[\d.\s]+", "", last.lower())
    if any(last.startswith(k) for k in _BIB_HEADINGS):
        return True
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return len(lines) >= 2 and len(_BIB_LINE.findall(text or "")) / len(lines) >= 0.6


@dataclass
class Evidence:
    document_id: int
    chunk_index: int
    page: Optional[int]
    headings: List[str]
    text: str
    score: float
    via: List[str] = field(default_factory=list)       # shared entities, if graph-linked
    figures: List[dict] = field(default_factory=list)  # figures linked to this chunk
    sid: str = ""  # source id such as "S3": assigned by the caller, never by this module
    demoted: bool = False  # reference-list chunk whose score was penalised (not serialised)

    @property
    def key(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "headings": self.headings,
            "text": self.text,
            "score": round(self.score, 4),
            "linked_via": self.via,
            "figures": self.figures,
        }


@dataclass
class SubQuestion:
    q: str
    document_ids: List[int]


# --------------------------------------------------------------- queries

def parse_queries(raw: Any, allowed: List[int]) -> List[SubQuestion]:
    """Validate LLM output. The model may never widen scope beyond `allowed`."""
    out: List[SubQuestion] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = str(item.get("q", "")).strip()
        if not q:
            continue
        ids = []
        for d in item.get("document_ids") or []:
            try:
                d = int(d)
            except (TypeError, ValueError):
                continue
            if d in allowed and d not in ids:
                ids.append(d)
        out.append(SubQuestion(q=q, document_ids=ids or list(allowed)))
    return out


# --------------------------------------------------------------- retrieval

def collect(
    subs: List[SubQuestion],
    store: Dict[Tuple[int, int], Evidence],
    trace: list,
    allowed_doc_ids: List[int],
) -> int:
    """Run each sub-question against each target document (vector + neighbour chunks),
    then follow shared entities to related chunks in ANY allowed document.
    Returns # new chunks."""
    from app.services.embedding_service import search_with_graph
    from app.services.graph_service import chunk_key, find_chunks_via_entities

    new_chunks = 0
    for sub in subs:
        wants_refs = bool(_ASKS_ABOUT_REFS.search(sub.q))
        direct_keys: List[str] = []
        for doc_id in sub.document_ids:
            hits = search_with_graph(
                query=sub.q, limit=HITS_PER_QUERY, document_id=doc_id
            )
            added = 0
            for h in hits:
                idx = h.get("chunk_index")
                if idx is None:
                    continue
                key = (doc_id, idx)
                score = float(h.get("score") or 0.0)
                bibliography = not wants_refs and is_bibliography(
                    h.get("headings"), h.get("text")
                )
                if bibliography:
                    score *= BIBLIOGRAPHY_PENALTY
                # real vector hit (not a neighbour); reference lists never seed the
                # entity hop, since author names would link unrelated chunks
                if h.get("source") != "graph" and not bibliography:
                    direct_keys.append(chunk_key(doc_id, idx))
                if key in store:
                    store[key].score = max(store[key].score, score)
                    continue
                store[key] = Evidence(
                    document_id=doc_id,
                    chunk_index=idx,
                    page=h.get("page"),
                    headings=h.get("headings") or [],
                    text=h.get("text") or "",
                    score=score,
                    demoted=bibliography,
                )
                added += 1
            new_chunks += added
            trace.append({
                "step": "retrieve",
                "query": sub.q,
                "document_id": doc_id,
                "hits": len(hits),
                "new": added,
            })

        # Graph hop: chunks that share entities with this sub-question's direct hits.
        try:
            linked = find_chunks_via_entities(direct_keys, allowed_doc_ids, limit=6)
        except Exception as e:  # Neo4j down must not kill the answer
            logger.warning(f"Entity expansion failed: {e}")
            linked = []
        added = 0
        for row in linked:
            key = (int(row["document_id"]), row["idx"])
            if key in store:
                continue
            # Ranked below direct vector hits; more shared entities -> slightly higher.
            store[key] = Evidence(
                document_id=key[0],
                chunk_index=row["idx"],
                page=row.get("page"),
                headings=row.get("headings") or [],
                text=row.get("text") or "",
                score=min(0.75, 0.5 + 0.05 * row.get("shared", 1)),
                via=row.get("via") or [],
            )
            added += 1
        new_chunks += added
        trace.append({
            "step": "entity_expand",
            "query": sub.q,
            "linked": len(linked),
            "new": added,
            "via": sorted({v for r in linked for v in (r.get("via") or [])})[:10],
        })
    return new_chunks

def select_evidence(store: Dict[Tuple[int, int], Evidence]) -> List[Evidence]:
    """Top-K per document (so one document can't crowd out the others), then
    ordered by document and position so the context reads naturally."""
    by_doc: Dict[int, List[Evidence]] = {}
    for e in store.values():
        by_doc.setdefault(e.document_id, []).append(e)

    chosen: List[Evidence] = []
    for items in by_doc.values():
        # Reference lists only fill the context when there is almost nothing else.
        real = [e for e in items if not e.demoted]
        if len(real) >= MIN_REAL_EVIDENCE:
            items = real
        items.sort(key=lambda e: e.score, reverse=True)
        chosen.extend(items[:EVIDENCE_PER_DOC])

    chosen.sort(key=lambda e: (e.document_id, e.chunk_index))
    return chosen


def attach_figures(evidence: List[Evidence], trace: list) -> None:
    """Look up figures linked (caption / in-text mention / nearby) to the chosen chunks."""
    from app.services.graph_service import get_figures_for_chunks

    by_key = {e.key: e for e in evidence}
    try:
        rows = get_figures_for_chunks(list(by_key))
    except Exception as e:  # figures are an enhancement; never break retrieval
        logger.warning(f"Figure lookup failed: {e}")
        return

    seen = set()
    for r in rows:
        owner = by_key.get(r["chunk_key"])
        if owner is None:
            continue
        fk = (owner.key, r["document_id"], r["figure_index"])
        if fk in seen:
            continue
        seen.add(fk)
        owner.figures.append({
            "document_id": r["document_id"],
            "figure_index": r["figure_index"],
            "label": r.get("label"),
            "number": r.get("number"),
            "caption": r.get("caption") or "",
            "page": r.get("page"),
            "relation": r.get("rel"),
            "image_url": f"/documents/{r['document_id']}/figures/{r['figure_index']}/image",
        })
    trace.append({"step": "figures", "linked": len(seen)})


def retrieve(queries: List[dict], allowed_doc_ids: List[int]) -> Dict[str, Any]:
    """Stateless retrieval for an external orchestrator (the Laravel agent loop).

    queries: [{"q": "...", "document_ids": [..optional..]}]; document ids are intersected
    with allowed_doc_ids, so a caller can never reach outside its own documents.
    The caller merges results across calls by `key` and assigns its own source ids."""
    allowed = [int(d) for d in allowed_doc_ids]
    subs = parse_queries(queries, allowed)[:MAX_QUERIES_PER_CALL]
    if not subs:
        raise ValueError("no valid queries")

    trace: List[Dict[str, Any]] = []
    store: Dict[Tuple[int, int], Evidence] = {}
    collect(subs, store, trace, allowed)
    evidence = select_evidence(store)
    attach_figures(evidence, trace)
    return {"evidence": [e.to_dict() for e in evidence], "trace": trace}
