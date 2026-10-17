"""
Agentic question answering over one or more documents.

Flow:
    plan      -> split the question into sub-questions (each targeting documents)
    retrieve  -> hybrid vector + graph-neighbour search, per sub-question, per document
    grade     -> LLM decides if the evidence is enough; if not, proposes follow-up queries
    (repeat retrieve/grade up to MAX_ITERATIONS)
    synthesize-> answer using ONLY numbered sources, citing them as [S1], [S2]
    verify    -> strip any citation that does not map to a retrieved source

Every step is appended to `trace`, so the caller can show how the answer was reached.
"""
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from app.services import llm_service

logger = logging.getLogger(__name__)

MAX_SUBQUESTIONS = int(os.getenv("AGENT_MAX_SUBQUESTIONS", 4))
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 3))
HITS_PER_QUERY = int(os.getenv("AGENT_HITS_PER_QUERY", 4))
EVIDENCE_PER_DOC = int(os.getenv("AGENT_EVIDENCE_PER_DOC", 6))
GRADE_SNIPPET_CHARS = 500

CITE_RE = re.compile(r"\[([^\[\]]*?S\d+[^\[\]]*?)\]")


@dataclass
class Evidence:
    document_id: int
    chunk_index: int
    page: Optional[int]
    headings: List[str]
    text: str
    score: float
    via: List[str] = field(default_factory=list)  # shared entities, if graph-linked
    figures: List[dict] = field(default_factory=list)  # figures linked to this chunk
    sid: str = ""  # assigned just before synthesis


@dataclass
class SubQuestion:
    q: str
    document_ids: List[int]


@dataclass
class AgentResult:
    answer: str
    citations: List[Dict[str, Any]]
    trace: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- planning

def _plan(question: str, doc_ids: List[int], trace: list) -> List[SubQuestion]:
    prompt = f"""You plan retrieval for a research assistant.
Available document ids: {doc_ids}

Break the user's question into at most {MAX_SUBQUESTIONS} short, self-contained
search queries. If the question compares or spans documents, create one query per
relevant document. If it is simple, return a single query.

Return JSON only:
{{"sub_questions": [{{"q": "<query>", "document_ids": [<ids from the list>]}}]}}

Question: {question}"""

    data = llm_service.chat_json(
        [{"role": "user", "content": prompt}],
        default={},
    )
    subs = _parse_subquestions(data.get("sub_questions"), doc_ids)
    if not subs:  # fallback: single pass over all docs
        subs = [SubQuestion(q=question, document_ids=list(doc_ids))]
    trace.append({"step": "plan", "sub_questions": [asdict(s) for s in subs]})
    return subs[:MAX_SUBQUESTIONS]


def _parse_subquestions(raw: Any, allowed: List[int]) -> List[SubQuestion]:
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

def _retrieve(
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
                if h.get("source") != "graph":  # real vector hit, not a neighbour
                    direct_keys.append(chunk_key(doc_id, idx))
                score = float(h.get("score") or 0.0)
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


# ----------------------------------------------------------------- grading

def _grade(
    question: str,
    doc_ids: List[int],
    store: Dict[Tuple[int, int], Evidence],
    trace: list,
) -> List[SubQuestion]:
    """Return follow-up sub-questions; empty list means evidence is sufficient."""
    ranked = sorted(store.values(), key=lambda e: e.score, reverse=True)[:15]
    if not ranked:
        trace.append({"step": "grade", "sufficient": False, "reason": "no evidence"})
        return []  # nothing more a rewrite can do blindly; synthesis will say so

    listing = "\n\n".join(
        f"(doc {e.document_id}, p.{e.page}) {e.text[:GRADE_SNIPPET_CHARS]}"
        for e in ranked
    )
    prompt = f"""You judge whether retrieved passages are enough to answer a question.

Question: {question}
Available document ids: {doc_ids}

Passages:
{listing}

If the passages contain what is needed (for every document the question concerns),
answer sufficient=true. Otherwise list up to 3 NEW, differently-worded search queries
that target what is missing.

Return JSON only:
{{"sufficient": true|false, "missing": [{{"q": "<query>", "document_ids": [<ids>]}}]}}"""

    data = llm_service.chat_json(
        [{"role": "user", "content": prompt}],
        default={"sufficient": True},  # on LLM failure, stop looping
    )
    if data.get("sufficient", True):
        trace.append({"step": "grade", "sufficient": True})
        return []

    follow_ups = _parse_subquestions(data.get("missing"), doc_ids)[:3]
    trace.append({
        "step": "grade",
        "sufficient": False,
        "follow_ups": [asdict(s) for s in follow_ups],
    })
    return follow_ups


# ---------------------------------------------------------------- synthesis

def _select_evidence(store: Dict[Tuple[int, int], Evidence]) -> List[Evidence]:
    """Top-K per document (so one document can't crowd out the others), then
    ordered by document and position so the context reads naturally."""
    by_doc: Dict[int, List[Evidence]] = {}
    for e in store.values():
        by_doc.setdefault(e.document_id, []).append(e)

    chosen: List[Evidence] = []
    for doc_id, items in by_doc.items():
        items.sort(key=lambda e: e.score, reverse=True)
        chosen.extend(items[:EVIDENCE_PER_DOC])

    chosen.sort(key=lambda e: (e.document_id, e.chunk_index))
    for i, e in enumerate(chosen, start=1):
        e.sid = f"S{i}"
    return chosen


def _label(e: Evidence) -> str:
    parts = [f"document {e.document_id}"]
    if e.page is not None:
        parts.append(f"page {e.page}")
    if e.headings:
        parts.append("section: " + " > ".join(e.headings))
    if e.via:
        parts.append("related through shared concepts: " + ", ".join(e.via))
    return ", ".join(parts)


def _attach_figures(evidence: List[Evidence], trace: list) -> None:
    """Look up figures linked (caption / in-text mention / nearby) to the chosen chunks."""
    from app.services.graph_service import chunk_key, get_figures_for_chunks

    by_key = {chunk_key(e.document_id, e.chunk_index): e for e in evidence}
    try:
        rows = get_figures_for_chunks(list(by_key))
    except Exception as e:  # figures are an enhancement; never break the answer
        logger.warning(f"Figure lookup failed: {e}")
        return

    seen = set()
    for r in rows:
        owner = by_key.get(r["chunk_key"])
        if owner is None:
            continue
        fk = (owner.sid, r["document_id"], r["figure_index"])
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


def _figure_lines(e: Evidence) -> str:
    if not e.figures:
        return ""
    lines = []
    for f in e.figures:
        name = f["label"] or "Unlabeled figure"
        where = f" (page {f['page']})" if f.get("page") is not None else ""
        cap = f": {f['caption'][:200]}" if f.get("caption") else ""
        lines.append(f"  - {name}{where}{cap}")
    return "\nLinked figures:\n" + "\n".join(lines)


def _verify_figures(answer: str, evidence: List[Evidence]) -> List[str]:
    """Every 'Figure N' the answer mentions must exist in the retrieved material,
    either as a linked figure or in a source's own text. Flags (does not rewrite)
    anything else, because a fabricated figure reference is a hard failure."""
    from app.services.figure_service import extract_figure_numbers

    allowed = set()
    for e in evidence:
        allowed |= extract_figure_numbers(e.text)
        for f in e.figures:
            if f.get("number"):
                allowed.add(str(f["number"]))

    unverified = sorted(extract_figure_numbers(answer) - allowed, key=int)
    if not unverified:
        return []
    return [
        "Answer mentions figure(s) not found in the retrieved sources: "
        + ", ".join(f"Figure {n}" for n in unverified)
    ]


def _synthesize(question: str, evidence: List[Evidence]) -> str:
    sources = "\n\n".join(
        f"[{e.sid}] ({_label(e)})\n{e.text}{_figure_lines(e)}" for e in evidence
    )
    system = (
        "You answer research questions strictly from the provided sources. "
        "Rules: (1) Use only facts stated in the sources. (2) After every claim, cite "
        "the supporting source ids like [S2] or [S1][S4]. (3) When documents differ, "
        "say which document each statement comes from. (4) Mention a figure only if it "
        "is listed under 'Linked figures' of a source or named in that source's text, "
        "and cite that source; never invent figure numbers, tables or pages. "
        "(5) If the sources do not "
        "contain the answer, say so plainly instead of guessing."
    )
    user = f"Sources:\n\n{sources}\n\nQuestion: {question}\n\nAnswer:"
    return llm_service.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
    ).strip()


# ------------------------------------------------------------- verification

def _verify(answer: str, evidence: List[Evidence]) -> Tuple[str, List[Evidence], List[str]]:
    """Remove citations that don't map to a retrieved source; report what happened."""
    valid = {e.sid for e in evidence}
    warnings: List[str] = []
    invalid_seen: set = set()

    def fix(match: re.Match) -> str:
        ids = re.findall(r"S\d+", match.group(1))
        good = [i for i in ids if i in valid]
        invalid_seen.update(i for i in ids if i not in valid)
        return "".join(f"[{i}]" for i in good)

    cleaned = CITE_RE.sub(fix, answer)

    if invalid_seen:
        warnings.append(
            f"Removed citations to non-existent sources: {sorted(invalid_seen)}"
        )

    cited_ids = set(re.findall(r"\[(S\d+)\]", cleaned))
    if not cited_ids:
        warnings.append("Answer contains no verifiable citations.")

    cited = [e for e in evidence if e.sid in cited_ids]
    return cleaned, cited, warnings


# --------------------------------------------------------------- public API

def answer_question(question: str, document_ids: List[int]) -> AgentResult:
    if not document_ids:
        raise ValueError("document_ids must not be empty")

    trace: List[Dict[str, Any]] = []
    store: Dict[Tuple[int, int], Evidence] = {}

    pending = _plan(question, document_ids, trace)

    for iteration in range(1, MAX_ITERATIONS + 1):
        trace.append({"step": "iteration", "n": iteration})
        new_chunks = _retrieve(pending, store, trace, document_ids)

        # Stop if a follow-up round brought nothing new, or this was the last round.
        if iteration > 1 and new_chunks == 0:
            trace.append({"step": "stop", "reason": "no new evidence"})
            break
        if iteration == MAX_ITERATIONS:
            break

        pending = _grade(question, document_ids, store, trace)
        if not pending:
            break

    evidence = _select_evidence(store)
    if not evidence:
        return AgentResult(
            answer="I couldn't find any relevant content in the selected documents.",
            citations=[],
            trace=trace,
            warnings=["No evidence retrieved."],
        )

    _attach_figures(evidence, trace)
    raw_answer = _synthesize(question, evidence)
    answer, cited, warnings = _verify(raw_answer, evidence)
    warnings += _verify_figures(answer, evidence)
    trace.append({"step": "synthesize", "sources_in_context": len(evidence),
                  "sources_cited": len(cited)})

    citations = [
        {
            "id": e.sid,
            "document_id": e.document_id,
            "page": e.page,
            "headings": e.headings,
            "chunk_index": e.chunk_index,
            "quote": e.text[:300],
            "figures": e.figures,
            "linked_via": e.via,
        }
        for e in cited
    ]
    return AgentResult(answer=answer, citations=citations, trace=trace, warnings=warnings)