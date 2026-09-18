from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticChunker:
    """
    Simple structure-aware semantic chunker.
    """

    ATOMIC_TYPES = {
        "code",
        "picture",
        "table",
    }

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.70,
        max_tokens: int = 200,
        min_tokens: int = 80,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.tokenizer = self.model.tokenizer

    # ============================================================
    # PUBLIC
    # ============================================================

    def chunk(
        self,
        elements: Sequence[Dict[str, Any]],
        document_id: int | None = None,
    ) -> List[Dict[str, Any]]:

        if not elements:
            return []

        elements = self._prepare_elements(elements)
        if not elements:
            return []

        texts = [element["text"] for element in elements]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        chunks = []
        current_elements = []
        current_tokens = 0

        for index, element in enumerate(elements):
            text = element["text"]
            tokens = self._count_tokens(text)

            if not current_elements:
                current_elements.append(element)
                current_tokens = tokens
                continue

            previous = current_elements[-1]

            # Never mix different sections
            if self._section_path(previous) != self._section_path(element):
                chunks.append(
                    self._build_chunk(current_elements, len(chunks), document_id)
                )
                current_elements = [element]
                current_tokens = tokens
                continue

            # Atomic element
            if self._is_atomic(element):
                chunks.append(
                    self._build_chunk(current_elements, len(chunks), document_id)
                )
                chunks.append(
                    self._build_chunk([element], len(chunks), document_id)
                )
                current_elements = []
                current_tokens = 0
                continue

            # Current chunk contains atomic element
            if any(self._is_atomic(item) for item in current_elements):
                chunks.append(
                    self._build_chunk(current_elements, len(chunks), document_id)
                )
                current_elements = [element]
                current_tokens = tokens
                continue

            # Maximum size
            if current_tokens + tokens > self.max_tokens:
                chunks.append(
                    self._build_chunk(current_elements, len(chunks), document_id)
                )
                current_elements = [element]
                current_tokens = tokens
                continue

            # Semantic similarity
            previous_index = index - 1
            similarity = self._cosine_similarity(
                embeddings[previous_index],
                embeddings[index],
            )

            if similarity < self.similarity_threshold:
                chunks.append(
                    self._build_chunk(current_elements, len(chunks), document_id)
                )
                current_elements = [element]
                current_tokens = tokens
            else:
                current_elements.append(element)
                current_tokens += tokens

        if current_elements:
            chunks.append(
                self._build_chunk(current_elements, len(chunks), document_id)
            )

        chunks = self._merge_small_chunks(chunks)

        for index, chunk in enumerate(chunks):
            chunk["chunk_index"] = index
            if document_id is not None:
                chunk["chunk_id"] = f"doc{document_id}_chunk_{index}"
            else:
                chunk["chunk_id"] = f"chunk_{index}"

        return chunks

    # ============================================================
    # PREPARE
    # ============================================================

    def _prepare_elements(self, elements):
        result = []
        for element in elements:
            text = (element.get("text") or "").strip()
            if not text:
                continue
            normalized = dict(element)
            normalized["text"] = text
            result.append(normalized)
        return result

    # ============================================================
    # BUILD CHUNK
    # ============================================================

    def _build_chunk(self, elements, chunk_index, document_id):
        texts = [element["text"] for element in elements]
        text = "\n\n".join(texts)
        section_path = self._section_path(elements[0])

        pages = []
        element_ids = []
        element_types = []

        for element in elements:
            page = element.get("page")
            if page is not None:
                pages.append(page)

            element_id = element.get("id")
            if element_id is not None:
                element_ids.append(str(element_id))

            element_type = element.get("type") or ""
            if element_type:
                element_types.append(element_type)

        return {
            "chunk_index": chunk_index,
            "chunk_id": (
                f"doc{document_id}_chunk_{chunk_index}"
                if document_id is not None
                else f"chunk_{chunk_index}"
            ),
            "text": text,
            "tokens": self._count_tokens(text),
            "section_path": section_path,
            "heading": section_path[-1] if section_path else None,
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "element_ids": element_ids,
            "element_types": list(dict.fromkeys(element_types)),
            "contains_code": "code" in element_types,
            "contains_picture": "picture" in element_types,
            "contains_table": "table" in element_types,
        }

    # ============================================================
    # MERGE SMALL CHUNKS (loop until stable)
    # ============================================================

    def _merge_small_chunks(self, chunks):
        if len(chunks) <= 1:
            return chunks

        previous_count = None
        while previous_count != len(chunks):
            previous_count = len(chunks)
            chunks = self._single_merge_pass(chunks)

        return chunks

    def _single_merge_pass(self, chunks):
        if len(chunks) <= 1:
            return chunks

        result = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            if i + 1 >= len(chunks):
                result.append(current)
                i += 1
                continue

            next_chunk = chunks[i + 1]

            # ----------------------------------------------------
            # Section compatibility (parent مشترک → merge مجاز)
            # ----------------------------------------------------
            if not self._section_compatible(
                current["section_path"],
                next_chunk["section_path"],
            ):
                result.append(current)
                i += 1
                continue

            # ----------------------------------------------------
            # Header-only: merge با بعدی
            # ----------------------------------------------------
            if self._is_header_only(current):
                combined_tokens = current["tokens"] + next_chunk["tokens"]
                if combined_tokens <= self.max_tokens:
                    merged = self._merge_chunks(current, next_chunk)
                    result.append(merged)
                    i += 2
                    continue

            # ----------------------------------------------------
            # Atomic ها رو merge نکن
            # ----------------------------------------------------
            if (
                current["contains_code"]
                or current["contains_picture"]
                or current["contains_table"]
                or next_chunk["contains_code"]
                or next_chunk["contains_picture"]
                or next_chunk["contains_table"]
            ):
                result.append(current)
                i += 1
                continue

            # ----------------------------------------------------
            # Current is big enough
            # ----------------------------------------------------
            if current["tokens"] >= self.min_tokens:
                result.append(current)
                i += 1
                continue

            # ----------------------------------------------------
            # Max size check
            # ----------------------------------------------------
            combined_tokens = current["tokens"] + next_chunk["tokens"]
            if combined_tokens > self.max_tokens:
                result.append(current)
                i += 1
                continue

            # ----------------------------------------------------
            # Merge
            # ----------------------------------------------------
            merged = self._merge_chunks(current, next_chunk)
            result.append(merged)
            i += 2

        return result

    # ============================================================
    # SECTION COMPATIBILITY
    # ============================================================

    @staticmethod
    def _section_compatible(path_a, path_b):
        """
        آیا دو section قابل merge هستن؟
        - برابر باشن → بله
        - یکی prefix اون یکی باشه → بله
        - parent مشترک (اولین آیتم) داشته باشن → بله
        - یکی خالی باشه → بله
        """
        if path_a == path_b:
            return True

        if path_a and path_b and path_a[0] == path_b[0]:
            return True

        shorter, longer = (
            (path_a, path_b) if len(path_a) <= len(path_b) else (path_b, path_a)
        )
        if len(shorter) == 0:
            return True
        return longer[: len(shorter)] == shorter

    # ============================================================
    # HEADER CHECK
    # ============================================================

    @staticmethod
    def _is_header_only(chunk):
        element_types = chunk.get("element_types") or []
        if not element_types:
            return False
        return element_types[0] == "section_header"

    # ============================================================
    # MERGE CHUNKS
    # ============================================================

    def _merge_chunks(self, a, b):
        merged_text = a["text"] + "\n\n" + b["text"]
        merged = dict(a)
        merged["text"] = merged_text
        merged["tokens"] = self._count_tokens(merged_text)

        # pages
        a_start = a.get("page_start")
        b_start = b.get("page_start")
        if a_start is not None and b_start is not None:
            merged["page_start"] = min(a_start, b_start)

        a_end = a.get("page_end")
        b_end = b.get("page_end")
        if a_end is not None and b_end is not None:
            merged["page_end"] = max(a_end, b_end)

        # element ids / types
        merged["element_ids"] = (
            (a.get("element_ids") or []) + (b.get("element_ids") or [])
        )
        merged["element_types"] = list(
            dict.fromkeys(
                (a.get("element_types") or []) + (b.get("element_types") or [])
            )
        )

        # flags
        merged["contains_code"] = (
            a.get("contains_code", False) or b.get("contains_code", False)
        )
        merged["contains_picture"] = (
            a.get("contains_picture", False) or b.get("contains_picture", False)
        )
        merged["contains_table"] = (
            a.get("contains_table", False) or b.get("contains_table", False)
        )

        # section_path: اگه متفاوت بودن، parent مشترک رو نگه دار
        a_path = a.get("section_path") or []
        b_path = b.get("section_path") or []

        if a_path != b_path:
            common = []
            for x, y in zip(a_path, b_path):
                if x == y:
                    common.append(x)
                else:
                    break
            if common:
                merged["section_path"] = common
                merged["heading"] = common[-1]
            else:
                # هیچ parent مشترکی نیست → طولانی‌تر رو نگه دار
                if len(b_path) > len(a_path):
                    merged["section_path"] = b_path
                    merged["heading"] = b.get("heading")
        else:
            if len(b_path) > len(a_path):
                merged["section_path"] = b_path
                merged["heading"] = b.get("heading")

        return merged

    # ============================================================
    # SIMILARITY
    # ============================================================

    @staticmethod
    def _cosine_similarity(vector_a, vector_b):
        return float(np.dot(vector_a, vector_b))

    # ============================================================
    # TOKEN COUNT
    # ============================================================

    def _count_tokens(self, text):
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    # ============================================================
    # SECTION
    # ============================================================

    @staticmethod
    def _section_path(element):
        path = element.get("section_path") or []
        if isinstance(path, list):
            return [str(item).strip() for item in path if str(item).strip()]
        if path:
            return [str(path).strip()]
        return []

    # ============================================================
    # ATOMIC
    # ============================================================

    def _is_atomic(self, element):
        element_type = (element.get("type") or "").lower()
        return element_type in self.ATOMIC_TYPES

    # ============================================================
    # STATS
    # ============================================================

    def get_stats(self, chunks):
        if not chunks:
            return {
                "total_chunks": 0,
                "total_tokens": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
            }

        token_counts = [c.get("tokens", 0) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "total_tokens": sum(token_counts),
            "avg_tokens": round(sum(token_counts) / len(chunks), 1),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "chunks_with_code": sum(1 for c in chunks if c.get("contains_code")),
            "chunks_with_table": sum(1 for c in chunks if c.get("contains_table")),
            "chunks_with_picture": sum(1 for c in chunks if c.get("contains_picture")),
        }