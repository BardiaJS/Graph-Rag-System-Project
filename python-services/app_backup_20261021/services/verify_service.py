"""
Deterministic answer verification. No LLM, no database: it only compares the answer text
with the sources the caller says it used. Backs POST /verify.

Why it lives here and not in the orchestrator: figure-reference parsing handles Persian
digits, "Figures 2, 3 and 5", sub-figure letters and year look-alikes (see figure_service),
and that logic should exist exactly once.

Source ids must look like S<number> (S1, S2, ...): that is what the answer is told to cite.
"""
import re
from typing import Any, Dict, List

from app.services.figure_service import extract_figure_numbers

CITE_RE = re.compile(r"\[([^\[\]]*?S\d+[^\[\]]*?)\]")


def verify_answer(answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    answer:  text containing citations such as [S1] or [S1][S4]
    sources: [{"sid": "S1", "text": "...", "figure_numbers": ["3", "10"]}]
             figure_numbers = numbers of figures linked to that source (from /retrieve)

    Returns:
        answer             citations to unknown sources removed
        cited_sids         valid source ids the answer actually cites, in numeric order
        unverified_figures figure numbers the answer mentions that no source supports
        warnings           human-readable problems (empty list = clean)
    A fabricated citation or figure reference is a hard failure, so citations are
    stripped and figure mentions are flagged (rewriting prose safely is not possible).
    """
    valid = {str(s["sid"]) for s in sources}
    warnings: List[str] = []
    invalid_seen = set()

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

    cited = sorted(set(re.findall(r"\[(S\d+)\]", cleaned)), key=_num)
    if not cited:
        warnings.append("Answer contains no verifiable citations.")

    # Every "Figure N" in the answer must be backed by a source: either a linked figure
    # or a mention in the source's own text.
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

    return {
        "answer": cleaned,
        "cited_sids": cited,
        "unverified_figures": unverified,
        "warnings": warnings,
    }


def _num(sid: str) -> int:
    return int(sid[1:])