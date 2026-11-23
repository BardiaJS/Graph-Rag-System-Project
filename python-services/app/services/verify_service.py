"""
Deterministic answer verification. No LLM, no database: it only compares the answer text
with the sources the caller says it used. Backs POST /verify.

What it checks (a fabricated citation or figure is a hard failure, so these matter):

  1. Citations to sources that do not exist are removed.
  2. A REFUSAL ("the sources do not cover this") correctly cites nothing: any citations the
     model attached to it are stripped.
  3. CITATION STUFFING: small models often end an answer with [S1][S2]...[S12]. When more than
     VERIFY_MAX_CITED sources are cited, only those whose text really matches the answer are kept.
  4. Figure references ("Figure 7") must exist in the sources or their linked figures.
  5. Numbers in the answer (years, ports, quantities) must appear in some source. A number found
     in none of them is the signature of an answer from the model's own memory, however many
     sources it cites. This is flagged, not rewritten.
  6. If most distinctive words of the answer appear in no source, that is flagged too.

Source ids must look like S<number> (S1, S2, ...): that is what the answer is told to cite.
"""
import os
import re
import json
from typing import Any, Dict, List, Set

from app.services.figure_service import _DIGITS, MENTION_RE, extract_figure_numbers
from app.services.llm_service import chat_json

SEMANTIC_ENABLED = os.getenv("VERIFY_SEMANTIC_ENABLED", "false").lower() == "true"
SEMANTIC_MODEL = os.getenv("VERIFY_SEMANTIC_MODEL", "llama3.2")
SEMANTIC_TIMEOUT = int(os.getenv("VERIFY_SEMANTIC_TIMEOUT", 60))

CITATION_REPAIR_ENABLED = os.getenv("VERIFY_CITATION_REPAIR_ENABLED", "false").lower() == "true"
CITATION_REPAIR_MODEL = os.getenv("VERIFY_CITATION_REPAIR_MODEL", "llama3.2")
CITATION_REPAIR_TIMEOUT = int(os.getenv("VERIFY_CITATION_REPAIR_TIMEOUT", 60))
CITATION_REPAIR_MAX_TOKENS = int(os.getenv("VERIFY_CITATION_REPAIR_MAX_TOKENS", 512))


CITE_RE = re.compile(r"\[([^\[\]]*?S\d+[^\[\]]*?)\]")
MAX_CITED = int(os.getenv("VERIFY_MAX_CITED", 5))
MIN_SUPPORT_RATIO = float(os.getenv("VERIFY_MIN_SUPPORT_RATIO", 0.3))

# The refusal must be in the FIRST sentence of a short answer, so that
# "PHP does X [S1]. The sources do not mention Y." keeps its legitimate citation.
REFUSAL_RE = re.compile(
    r"(not (contain|mention|provide|include|found|covered)|no (relevant )?information|"
    r"cannot (find|answer)|can't (find|answer)|couldn.?t find|could not find|"
    r"do(es)? not (say|specify|discuss|address)|not (available )?in the (provided )?(sources|documents)|"
    r"(اطلاعاتی|پاسخی).{0,40}(وجود ندارد|یافت نشد|ارائه نشده)|نمی.?توان)",
    re.IGNORECASE,
)
REFUSAL_MAX_CHARS = 450

_STOP = frozenset(
    "that this with from have which their there about would could should into when what where also "
    "used using each more than such these those because other only some will then them they were "
    "been being both does done make most over same very your while after before between through "
    "however therefore source sources provided answer question document documents paper".split()
)
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _tokens(text: str) -> Set[str]:
    out = set()
    for w in re.findall(r"[^\W_]+", text or ""):
        w = w.lower()
        if len(w) >= (4 if w.isascii() else 3) and w not in _STOP and not w.isdigit():
            out.add(w)
    return out


def _numbers(text: str, answer_side: bool) -> Set[str]:
    """Numbers as normalised strings. On the answer side, citation markers and 'Figure N'
    mentions are removed first, and single digits (list numbering) are ignored."""
    t = (text or "").translate(_DIGITS)
    if answer_side:
        t = MENTION_RE.sub(" ", CITE_RE.sub(" ", t))
    nums = set()
    for m in _NUM.findall(t):
        n = m.replace(",", "")
        if not answer_side or len(re.sub(r"\D", "", n)) >= 2:
            nums.add(n)
    return nums


def _strip_markers(text: str, sids: Set[str]) -> str:
    """Remove [S#] markers for the given source ids, then tidy the spacing they leave."""
    def drop(match: re.Match) -> str:
        kept = [i for i in re.findall(r"S\d+", match.group(1)) if i not in sids]
        return "".join(f"[{i}]" for i in kept)

    out = CITE_RE.sub(drop, text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return re.sub(r"\s+([.,;:!?])", r"\1", out).strip()


def _prune(answer: str, cited: List[str], by_sid: Dict[str, str]) -> List[str]:
    """Keep the cited sources whose text actually overlaps the answer's content words."""
    answer_words = _tokens(CITE_RE.sub(" ", answer))
    scores = {sid: len(answer_words & _tokens(by_sid.get(sid, ""))) for sid in cited}
    best = max(scores.values(), default=0)
    if best == 0:
        return cited[:1]  # nothing matches lexically: keep one rather than none
    threshold = max(2, 0.4 * best)
    keep = [sid for sid in sorted(cited, key=lambda s: -scores[s]) if scores[sid] >= threshold]
    return keep[:MAX_CITED] or [max(cited, key=lambda s: scores[s])]

def _extract_claims(answer: str) -> List[Dict[str, Any]]:
    """
    Split an answer into sentences and associate each sentence
    with the citations appearing in that sentence.
    """
    parts = re.split(r"(?<=[.!?؟])\s+", answer.strip())
    claims = []

    for part in parts:
        text = part.strip()

        if not text:
            continue

        citations = sorted(
            set(re.findall(r"\[(S\d+)\]", text)),
            key=_num,
        )

        claim = CITE_RE.sub(" ", text)
        claim = re.sub(r"\s+", " ", claim).strip()

        claim = re.sub(r"\s+([,.!?؟])", r"\1", claim)

        if not claim or not re.search(r"\w", claim):
            continue

        claims.append({
            "index": len(claims),
            "claim": claim,
            "citations": citations,
            "raw": text,
        })

    return claims


def _repair_missing_citations(
    answer: str,
    by_sid: Dict[str, str],
) -> Dict[str, Any]:
    report = {
        "status": "disabled",
        "attempted": 0,
        "repaired": 0,
        "unresolved": 0,
        "reasons": [],
    }

    if not CITATION_REPAIR_ENABLED:
        return {"answer": answer, "report": report}

    claims = _extract_claims(answer)

    candidates = [
        item for item in claims
        if not item["citations"]
        and not item["claim"].startswith("#")
        and not re.search(r"[?؟]\\s*$", item["claim"])
        and not REFUSAL_RE.search(item["claim"])
    ]

    if not candidates:
        report["status"] = "no_candidates"
        return {"answer": answer, "report": report}

    report["status"] = "running"
    report["attempted"] = len(candidates)

    prompt_data = [
        {"index": item["index"], "claim": item["claim"]}
        for item in candidates
    ]

    source_data = [
        {"sid": sid, "text": source_text[:4000]}
        for sid, source_text in by_sid.items()
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict citation-grounding assistant. "
                "For each sentence, identify only sources that directly "
                "support its factual meaning. Use exclusively the supplied "
                "sources. Do not use outside knowledge. Similar keywords "
                "are not enough. Return the minimum sources needed. "
                "If no source directly supports a sentence, return an "
                "empty sids array. Never invent source IDs. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "SENTENCES:\n"
                + json.dumps(prompt_data, ensure_ascii=False)
                + "\n\nSOURCES:\n"
                + json.dumps(source_data, ensure_ascii=False)
                + '\n\nReturn exactly this JSON structure: '
                '{"repairs":[{"index":0,"sids":["S1"]}]} '
                "Include one item per sentence. Use [] if unsupported. "
                "Preserve indexes exactly."
            ),
        },
    ]

    try:
        result = chat_json(
            messages,
            default=None,
            strict=False,
            model=CITATION_REPAIR_MODEL,
            max_tokens=CITATION_REPAIR_MAX_TOKENS,
            timeout=CITATION_REPAIR_TIMEOUT,
        )
    except Exception as exc:
        result = None
        report["reasons"].append(str(exc)[:300])

    if not isinstance(result, dict) or not isinstance(result.get("repairs"), list):
        report["status"] = "model_error"
        report["unresolved"] = len(candidates)
        report["reasons"].append("Invalid repair model response.")
        return {"answer": answer, "report": report}

    candidate_indexes = {item["index"] for item in candidates}
    accepted = {}

    for item in result["repairs"]:
        if not isinstance(item, dict):
            continue

        index = item.get("index")
        if isinstance(index, str) and index.isdigit():
            index = int(index)

        if not isinstance(index, int) or index not in candidate_indexes:
            continue

        sids = item.get("sids")
        if not isinstance(sids, list):
            continue

        valid_sids = list(dict.fromkeys(
            sid for sid in sids
            if isinstance(sid, str) and sid in by_sid
        ))

        if valid_sids:
            accepted[index] = valid_sids[:3]

    repaired_answer = answer

    for item in candidates:
        sids = accepted.get(item["index"], [])
        if not sids:
            continue

        raw = item["raw"]
        citation = " " + "".join(f"[{sid}]" for sid in sids)

        match = re.search(r"([.!?؟])$", raw)
        updated = (
            raw[:match.start()] + citation + raw[match.start():]
            if match
            else raw + citation
        )

        position = repaired_answer.find(raw)
        if position == -1:
            continue

        repaired_answer = (
            repaired_answer[:position]
            + updated
            + repaired_answer[position + len(raw):]
        )
        report["repaired"] += 1

    report["unresolved"] = report["attempted"] - report["repaired"]
    report["status"] = "completed"

    if report["unresolved"]:
        report["reasons"].append(
            "Some sentences had no directly supporting source."
        )

    return {"answer": repaired_answer, "report": report}


def _semantic_verify(
    answer: str,
    by_sid: Dict[str, str],
) -> Dict[str, Any]:
    """
    Evaluate each answer claim only against its own cited sources.
    This is shadow evaluation: it never modifies the answer.
    """
    claims = _extract_claims(answer)

    if not claims:
        return {
            "status": "skipped",
            "checks": [],
        }

    payload = []

    for item in claims:
        cited_sources = [
            {
                "sid": sid,
                "text": by_sid[sid][:5000],
            }
            for sid in item["citations"]
            if sid in by_sid
        ]

        if not item["citations"]:
            payload.append({
                **item,
                "status": "missing_citation",
                "sources": [],
            })
        elif not cited_sources:
            payload.append({
                **item,
                "status": "unsupported",
                "sources": [],
            })
        else:
            payload.append({
                **item,
                "sources": cited_sources,
            })

    to_evaluate = [
        item for item in payload
        if "status" not in item
    ]

    if not to_evaluate:
        return {
            "status": "completed",
            "checks": [
                {
                    "claim": item["claim"],
                    "citations": item["citations"],
                    "status": item.get("status", "not_evaluated"),
                    "reason": (
                        "The claim has no citation."
                        if item.get("status") == "missing_citation"
                        else "No valid cited source was available."
                    ),
                }
                for item in payload
            ],
        }

    prompt_data = [
        {
            "index": item["index"],
            "claim": item["claim"],
            "sources": item["sources"],
        }
        for item in to_evaluate
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict citation-grounding evaluator. "
                "Evaluate whether EACH claim is supported by ONLY the "
                "sources explicitly attached to that claim. "
                "Do not use outside knowledge. "
                "Do not use other claims' sources. "
                "A source mentioning similar words is not sufficient: "
                "it must support the actual meaning of the claim. "
                "Distinguish direct support from partial support. "
                "Return JSON only."
                "A claim may contain multiple factual parts. "
                "If at least one important part is supported but another "
                "important part is not supported, classify it as "
                "partially_supported. "
                "Use unsupported only when none of the meaningful factual "
                "parts are supported by the cited sources. "
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate these claims and their cited sources.\n\n"
                f"{json.dumps(prompt_data, ensure_ascii=False)}\n\n"
                "Return this JSON structure:\n"
                "{"
                '"checks": ['
                "{"
                '"index": 0,'
                '"status": "supported",'
                '"reason": "Brief explanation"'
                "}"
                "]"
                "}\n\n"
                "Allowed statuses: supported, partially_supported, unsupported."
            ),
        },
    ]

    try:
        result = chat_json(
            messages,
            default=None,
            strict=True,
            model=SEMANTIC_MODEL,
            timeout=SEMANTIC_TIMEOUT,
        )
        print("SEMANTIC EVALUATOR RAW:", repr(result), flush=True)
        if not isinstance(result, dict) or not isinstance(result.get("checks"), list):
            raise ValueError("Invalid semantic verification response.")

        evaluated = {}

        expected_indexes = {
            item["index"] for item in to_evaluate
        }

        for check in result["checks"]:
            if not isinstance(check, dict):
                continue

            index = check.get("index")

            if isinstance(index, str) and index.isdigit():
                index = int(index)

            status = check.get("status")

            if (
                isinstance(index, int)
                and index in expected_indexes
                and status in {
                    "supported",
                    "partially_supported",
                    "unsupported",
                }
            ):
                evaluated[index] = {
                    "status": status,
                    "reason": str(check.get("reason", ""))[:500],
                }

        missing_items = [
            item for item in to_evaluate
            if item["index"] not in evaluated
        ]

        if missing_items:
            retry_index_map = {
                index: item["index"]
                for index, item in enumerate(missing_items)
            }

            retry_data = [
                {
                    "index": index,
                    "claim": item["claim"],
                    "sources": item["sources"],
                }
                for index, item in enumerate(missing_items)
            ]

            retry_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict citation evaluator. "
                        "Evaluate EVERY claim in the input. "
                        "Do not skip any claim. "
                        "Use only the sources attached to that claim. "
                        "Return exactly one check for every input index. "
                        "Allowed statuses: supported, partially_supported, unsupported. "
                        "Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Evaluate all of these claims:\n"
                        f"{json.dumps(retry_data, ensure_ascii=False)}\n"
                        'Return one check for every input index. '
                        'Preserve each index exactly as provided. '
                        'Do not renumber or invent indexes. '
                        'Example format: {"checks":[{"index":0,'
                        '"status":"supported","reason":"..."}]}'
                    ),
                },
            ]

            retry_result = chat_json(
                retry_messages,
                default=None,
                strict=True,
                model=SEMANTIC_MODEL,
                timeout=SEMANTIC_TIMEOUT,
            )

            print("SEMANTIC RETRY RAW:", repr(retry_result), flush=True)

            if isinstance(retry_result, dict) and isinstance(
                retry_result.get("checks"), list
            ):
                for check in retry_result["checks"]:
                    if not isinstance(check, dict):
                        continue

                    retry_index = check.get("index")

                    if isinstance(retry_index, str) and retry_index.isdigit():
                        retry_index = int(retry_index)

                    status = check.get("status")

                    if (
                        isinstance(retry_index, int)
                        and retry_index in retry_index_map
                        and status in {
                            "supported",
                            "partially_supported",
                            "unsupported",
                        }
                    ):
                        original_index = retry_index_map[retry_index]

                        evaluated[original_index] = {
                            "status": status,
                            "reason": str(check.get("reason", ""))[:500],
                        }

        final_checks = []

        for item in payload:
            if "status" in item:
                status = item["status"]
                reason = (
                    "The claim has no citation."
                    if status == "missing_citation"
                    else "No valid cited source was available."
                )
            else:
                evaluation = evaluated.get(item["index"])

                if evaluation:
                    status = evaluation["status"]
                    reason = evaluation["reason"]
                else:
                    status = "evaluation_error"
                    reason = "The evaluator did not return a valid result."

            final_checks.append({
                "claim": item["claim"],
                "citations": item["citations"],
                "status": status,
                "reason": reason,
            })

        return {
            "status": "completed",
            "checks": final_checks,
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc)[:500],
            "checks": [
                {
                    "claim": item["claim"],
                    "citations": item["citations"],
                    "status": "evaluation_error",
                    "reason": "Semantic evaluation failed.",
                }
                for item in payload
            ],
        }

def verify_answer(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    answer:  text containing citations such as [S1] or [S1][S4]
    sources: [{"sid": "S1", "text": "...", "figure_numbers": ["3", "10"]}]

    Returns answer (cleaned), cited_sids, pruned_sids, refusal, unverified_figures,
    unsupported_numbers and warnings (empty = clean).
    """
    by_sid = {str(s["sid"]): (s.get("text") or "") for s in sources}
    valid = set(by_sid)
    warnings: List[str] = []
    invalid_seen: Set[str] = set()

    def fix(match: re.Match) -> str:
        ids = re.findall(r"S\d+", match.group(1))
        good = [i for i in ids if i in valid]
        invalid_seen.update(i for i in ids if i not in valid)
        return "".join(f"[{i}]" for i in good)

    cleaned = CITE_RE.sub(fix, answer)
    if invalid_seen:
        warnings.append(
            f"Removed citations to non-existent sources: {sorted(invalid_seen, key=_num)}"
        )

    result = {"pruned_sids": [], "refusal": False, "unverified_figures": [], "unsupported_numbers": []}

    # 2) a refusal cites nothing
    plain = CITE_RE.sub("", cleaned).strip()
    first_sentence = re.split(r"(?<=[.!?؟])\s", plain, maxsplit=1)[0]
    if len(plain) <= REFUSAL_MAX_CHARS and REFUSAL_RE.search(first_sentence):
        return {
            **result,
            "answer": _strip_markers(cleaned, set(valid)),
            "cited_sids": [],
            "refusal": True,
            "warnings": warnings,
        }

    # 3) Repair missing citations before final citation checks.
    repair_result = _repair_missing_citations(cleaned, by_sid)
    cleaned = repair_result["answer"]
    citation_repair = repair_result["report"]

    cited = sorted(set(re.findall(r"\[(S\d+)\]", cleaned)), key=_num)

    # 3) citation stuffing
    if len(cited) > MAX_CITED:
        keep = _prune(cleaned, cited, by_sid)
        dropped = set(cited) - set(keep)
        cleaned = _strip_markers(cleaned, dropped)
        result["pruned_sids"] = sorted(dropped, key=_num)
        cited = sorted(keep, key=_num)

    if not cited:
        warnings.append("Answer contains no verifiable citations.")

    # 4) figure references
    allowed = set()
    for s in sources:
        allowed |= extract_figure_numbers(s.get("text") or "")
        allowed |= {str(n) for n in (s.get("figure_numbers") or []) if n}
    unverified = sorted(extract_figure_numbers(cleaned) - allowed, key=int)
    if unverified:
        warnings.append(
            "Answer mentions figure(s) not found in the retrieved sources: "
            + ", ".join(f"Figure {n}" for n in unverified)
        )
    result["unverified_figures"] = unverified

    # 5) numbers that no source contains
    source_numbers: Set[str] = set()
    for text in by_sid.values():
        source_numbers |= _numbers(text, answer_side=False)
    unsupported = sorted(_numbers(cleaned, answer_side=True) - source_numbers)
    if unsupported:
        warnings.append(
            "Answer contains number(s) that appear in none of the sources: " + ", ".join(unsupported)
        )
    result["unsupported_numbers"] = unsupported

    # 6) mostly unfamiliar vocabulary
    words = _tokens(CITE_RE.sub(" ", cleaned))
    if len(words) >= 6:
        known = set().union(*[_tokens(t) for t in by_sid.values()]) if by_sid else set()
        if len(words & known) / len(words) < MIN_SUPPORT_RATIO:
            warnings.append("Most terms in the answer do not appear in the sources.")

    # 7) optional semantic citation verification (shadow mode)
    semantic_checks = (
        _semantic_verify(cleaned, by_sid)
        if SEMANTIC_ENABLED
        else {
            "status": "disabled",
            "checks": [],
        }
    )
    return {
        **result,
        "answer": cleaned,
        "cited_sids": cited,
        "warnings": warnings,
        "semantic_checks": semantic_checks,
        "citation_repair": citation_repair,
    }

def _num(sid: str) -> int:
    return int(sid[1:])
