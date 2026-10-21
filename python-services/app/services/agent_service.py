"""
OPTIONAL server-side agent loop (POST /ask, disabled unless ENABLE_AGENT_ENDPOINT=1).

The intended home for this loop is the Laravel backend. The Python service exposes the
deterministic parts it needs, POST /retrieve and POST /verify, so this module is a working
reference implementation of the same loop, kept for prompts and for end-to-end testing.

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
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from app.services import llm_service
from app.services.retrieval_service import (
    Evidence,
    SubQuestion,
    attach_figures,
    collect,
    parse_queries,
    select_evidence,
)
from app.services.verify_service import verify_answer

logger = logging.getLogger(__name__)

MAX_SUBQUESTIONS = int(os.getenv("AGENT_MAX_SUBQUESTIONS", 4))
MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 3))
GRADE_SNIPPET_CHARS = 500







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
    subs = parse_queries(data.get("sub_questions"), doc_ids)
    if not subs:  # fallback: single pass over all docs
        subs = [SubQuestion(q=question, document_ids=list(doc_ids))]
    trace.append({"step": "plan", "sub_questions": [asdict(s) for s in subs]})
    return subs[:MAX_SUBQUESTIONS]




# --------------------------------------------------------------- retrieval



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

    follow_ups = parse_queries(data.get("missing"), doc_ids)[:3]
    trace.append({
        "step": "grade",
        "sufficient": False,
        "follow_ups": [asdict(s) for s in follow_ups],
    })
    return follow_ups


# ---------------------------------------------------------------- synthesis



def _label(e: Evidence) -> str:
    parts = [f"document {e.document_id}"]
    if e.page is not None:
        parts.append(f"page {e.page}")
    if e.headings:
        parts.append("section: " + " > ".join(e.headings))
    if e.via:
        parts.append("related through shared concepts: " + ", ".join(e.via))
    return ", ".join(parts)




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



# --------------------------------------------------------------- public API

def answer_question(question: str, document_ids: List[int]) -> AgentResult:
    if not document_ids:
        raise ValueError("document_ids must not be empty")

    trace: List[Dict[str, Any]] = []
    store: Dict[Tuple[int, int], Evidence] = {}

    pending = _plan(question, document_ids, trace)

    for iteration in range(1, MAX_ITERATIONS + 1):
        trace.append({"step": "iteration", "n": iteration})
        new_chunks = collect(pending, store, trace, document_ids)

        # Stop if a follow-up round brought nothing new, or this was the last round.
        if iteration > 1 and new_chunks == 0:
            trace.append({"step": "stop", "reason": "no new evidence"})
            break
        if iteration == MAX_ITERATIONS:
            break

        pending = _grade(question, document_ids, store, trace)
        if not pending:
            break

    evidence = select_evidence(store)
    for i, e in enumerate(evidence, start=1):
        e.sid = f"S{i}"  # source ids are an orchestration concern, assigned here
    if not evidence:
        return AgentResult(
            answer="I couldn't find any relevant content in the selected documents.",
            citations=[],
            trace=trace,
            warnings=["No evidence retrieved."],
        )

    attach_figures(evidence, trace)
    raw_answer = _synthesize(question, evidence)
    check = verify_answer(raw_answer, [
        {
            "sid": e.sid,
            "text": e.text,
            "figure_numbers": [f["number"] for f in e.figures if f.get("number")],
        }
        for e in evidence
    ])
    answer, warnings = check["answer"], check["warnings"]
    cited_ids = set(check["cited_sids"])
    cited = [e for e in evidence if e.sid in cited_ids]
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
