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
from typing import Any, Dict, List, Set

from app.services.figure_service import _DIGITS, MENTION_RE, extract_figure_numbers

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

    return {**result, "answer": cleaned, "cited_sids": cited, "warnings": warnings}


def _num(sid: str) -> int:
    return int(sid[1:])
