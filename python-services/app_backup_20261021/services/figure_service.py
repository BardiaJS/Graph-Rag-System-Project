"""
Figures: link extracted pictures to their caption chunk and to the paragraphs that
mention them, and store them in Neo4j.

Graph additions
    (Document)-[:HAS_FIGURE]->(Figure {key, document_id, figure_index, label, number,
                                       caption, page, bbox, image_file, width, height})
    (Chunk)-[:DESCRIBES]->(Figure)     the chunk that contains the figure's caption
    (Chunk)-[:REFERENCES]->(Figure)    chunks that mention "Figure N" in the text
    (Chunk)-[:NEAR]->(Figure)          fallback: first chunk on the same page, used only
                                       when no caption chunk could be matched

Images live on disk under FIGURES_DIR/<document_id>/fig_<index>.png. The path is always
built from integers, never from stored strings, so it cannot be steered by data.

Handles English ("Figure 3", "Fig. 3", "Figures 2 and 3") and Persian ("شکل ۳") labels.
"""
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

FIGURES_DIR = Path(os.getenv("FIGURES_DIR", "./data/figures"))

# Persian and Arabic-Indic digits -> ASCII
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

_WORD = r"(?:figures?|figs?|شکل)"
_ITEM = r"\d{1,3}(?!\d)[a-z]?"  # 1-3 digits (so a year like 2019 never matches) + subfigure letter

# "Figure 3", "Fig.3b", "Figures 2, 3 and 5"
MENTION_RE = re.compile(
    rf"\b{_WORD}\.?\s*({_ITEM}(?:\s*(?:,|and|&)\s*{_ITEM})*)", re.IGNORECASE
)
# Caption start: "Figure 3: ...", "Fig. 3. ...", "شکل ۳ - ..."
LABEL_RE = re.compile(rf"^\s*{_WORD}\.?\s*(\d{{1,3}})(?!\d)", re.IGNORECASE)


# ------------------------------------------------------------------- files

def figure_dir(document_id: int) -> Path:
    return FIGURES_DIR / str(int(document_id))


def figure_path(document_id: int, figure_index: int) -> Path:
    return figure_dir(document_id) / f"fig_{int(figure_index)}.png"


def purge_figure_files(document_id: int) -> None:
    shutil.rmtree(figure_dir(document_id), ignore_errors=True)


# ----------------------------------------------------------------- parsing

def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").translate(_DIGITS)).strip().lower()


def extract_figure_numbers(text: str) -> Set[str]:
    """All figure numbers mentioned in a text, as ASCII digit strings."""
    nums: Set[str] = set()
    for m in MENTION_RE.finditer((text or "").translate(_DIGITS)):
        nums.update(re.findall(r"\d+", m.group(1)))
    return nums


def parse_caption_number(caption: str) -> Optional[str]:
    m = LABEL_RE.match((caption or "").translate(_DIGITS))
    return m.group(1) if m else None


# ----------------------------------------------------------------- linking

def plan_figures(
    document_id: int, figures: List[dict], chunks: List[dict]
) -> Dict[str, List[dict]]:
    """Pure function: figure rows + chunk->figure link rows. Easy to unit-test."""
    chunk_texts = [_squash(c.get("text", "")) for c in chunks]
    chunk_numbers = [extract_figure_numbers(c.get("text", "")) for c in chunks]

    fig_rows: List[dict] = []
    links: List[dict] = []

    for fig in figures:
        i = int(fig["figure_index"])
        fkey = f"{document_id}:fig:{i}"
        caption = (fig.get("caption") or "").strip()
        number = parse_caption_number(caption)

        fig_rows.append({
            "key": fkey,
            "figure_index": i,
            "label": f"Figure {number}" if number else None,
            "number": number,
            "caption": caption,
            "page": fig.get("page"),
            "bbox": fig.get("bbox"),
            "image_file": fig.get("image_file"),
            "width": fig.get("width"),
            "height": fig.get("height"),
        })

        # 1) the chunk holding the caption text
        caption_chunk: Optional[int] = None
        probe = _squash(caption)[:80]
        if len(probe) >= 15:
            for idx, text in enumerate(chunk_texts):
                if probe in text:
                    caption_chunk = idx
                    break
        if caption_chunk is not None:
            links.append({"figure_key": fkey, "rel": "DESCRIBES",
                          "chunk_key": f"{document_id}:{caption_chunk}"})

        # 2) paragraphs that mention it ("see Figure 3")
        if number:
            for idx, nums in enumerate(chunk_numbers):
                if idx != caption_chunk and number in nums:
                    links.append({"figure_key": fkey, "rel": "REFERENCES",
                                  "chunk_key": f"{document_id}:{idx}"})

        # 3) fallback so an uncaptioned figure is still reachable
        if caption_chunk is None and fig.get("page") is not None:
            for idx, c in enumerate(chunks):
                if c.get("page") == fig.get("page"):
                    links.append({"figure_key": fkey, "rel": "NEAR",
                                  "chunk_key": f"{document_id}:{idx}"})
                    break

    return {"figures": fig_rows, "links": links}


# ------------------------------------------------------------------- Neo4j

def store_figures(
    document_id: int, figures: List[dict], chunks: List[dict]
) -> Dict[str, int]:
    """Write Figure nodes and their links. Call AFTER build_graph (needs Chunk nodes).
    build_graph/delete_document_graph already removes old Figure nodes of the document."""
    from app.services.graph_service import ensure_schema, get_driver

    document_id = int(document_id)
    plan = plan_figures(document_id, figures, chunks)
    if not plan["figures"]:
        return {"figures": 0, "links": 0, "unlinked": 0}

    ensure_schema()

    def tx(t):
        t.run(
            """
            UNWIND $rows AS row
            MERGE (f:Figure {key: row.key})
            SET f.document_id = $doc_id, f.figure_index = row.figure_index,
                f.label = row.label, f.number = row.number, f.caption = row.caption,
                f.page = row.page, f.bbox = row.bbox, f.image_file = row.image_file,
                f.width = row.width, f.height = row.height
            WITH f
            MATCH (d:Document {id: $doc_id})
            MERGE (d)-[:HAS_FIGURE]->(f)
            """,
            rows=plan["figures"], doc_id=document_id,
        )
        # relationship types can't be parameters: iterate a fixed whitelist
        for rel in ("DESCRIBES", "REFERENCES", "NEAR"):
            rows = [l for l in plan["links"] if l["rel"] == rel]
            if rows:
                t.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (c:Chunk {{key: row.chunk_key}})
                    MATCH (f:Figure {{key: row.figure_key}})
                    MERGE (c)-[:{rel}]->(f)
                    """,
                    rows=rows,
                )

    with get_driver().session() as session:
        session.execute_write(tx)

    linked = {l["figure_key"] for l in plan["links"]}
    stats = {
        "figures": len(plan["figures"]),
        "links": len(plan["links"]),
        "unlinked": len([f for f in plan["figures"] if f["key"] not in linked]),
    }
    logger.info(f"Figures stored for document {document_id}: {stats}")
    return stats