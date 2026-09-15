import re
from typing import Optional


class DoclingNormalizer:
    """
    عناصر خام DoclingInspector را به NormalizedDocument تبدیل می‌کند.

    خروجی:
      - elements: لیست تخت عناصر با section_path, parent_id, ...
      - sections: درخت سلسله‌مراتبی section ها
      - figures: لیست تصاویر
      - tables: لیست جدول‌ها
      - pages: لیست صفحات با element_ids
    """

    def __init__(self, elements: list, document_id: int = 0):
        self.elements = elements
        self.document_id = document_id
        self._normalized = None

    # ============================================================
    # PUBLIC API
    # ============================================================
    def normalize(self) -> dict:
        if self._normalized is not None:
            return self._normalized

        # ۱. split
        split_elements = self._split_broken_elements()

        # ۲. merge
        merged_elements = self._merge_fragments(split_elements)

        # ۳. section_path + parent_section
        with_paths = self._recompute_section_paths(merged_elements)

        # ۴. id + order
        for i, el in enumerate(with_paths, start=1):
            el["order"] = i
            el["id"] = self._make_id(el["page"], i)

        # ۵. درخت section ها
        sections = self._build_sections_tree(with_paths)

        # ۶. parent_id
        with_refs = self._add_tree_refs(with_paths, sections)

        # ۷. figures / tables / pages
        figures = self._extract_figures(with_refs)
        tables = self._extract_tables(with_refs)
        pages = self._extract_pages(with_refs)

        self._normalized = {
            "document_id": self.document_id,
            "title": self._get_title(with_refs),
            "stats": self._get_stats(with_refs),
            "elements": with_refs,
            "sections": sections,
            "figures": figures,
            "tables": tables,
            "pages": pages,
        }
        return self._normalized

    # ============================================================
    # SECTION PATHS
    # ============================================================
    def _recompute_section_paths(self, elements: list) -> list:
        stack: list[tuple[int, str]] = []

        for el in elements:
            if el["type"] == "section_header":
                level = el.get("level") or 1
                is_label = bool(
                    re.match(r"^(Syntax|Example)\s*:", el["text"], re.IGNORECASE)
                )

                if is_label:
                    el["section_path"] = [s[1] for s in stack]
                    el["section_level"] = len(stack)
                    el["parent_section"] = stack[-1][1] if stack else None
                else:
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    parent = stack[-1][1] if stack else None
                    stack.append((level, el["text"]))
                    el["section_path"] = [s[1] for s in stack]
                    el["section_level"] = level
                    el["parent_section"] = parent
            else:
                el["section_path"] = [s[1] for s in stack]
                el["section_level"] = len(stack)
                el["parent_section"] = stack[-1][1] if stack else None

        return elements

    # ============================================================
    # SECTIONS TREE
    # ============================================================
    def _build_sections_tree(self, elements: list) -> list:
        sections = []
        stack = []

        for el in elements:
            if el["type"] != "section_header":
                if stack:
                    stack[-1][0]["element_ids"].append(el["id"])
                continue

            level = el.get("level") or 1
            section = {
                "id": f"section_{el['id']}",
                "title": el["text"],
                "level": level,
                "page": el["page"],
                "element_id": el["id"],
                "parent_section_id": None,
                "children_section_ids": [],
                "element_ids": [],
                "section_path": el["section_path"],
            }

            while stack and stack[-1][1] >= level:
                stack.pop()

            if stack:
                parent = stack[-1][0]
                parent["children_section_ids"].append(section["id"])
                section["parent_section_id"] = parent["id"]

            sections.append(section)
            stack.append((section, level))

        return sections

    # ============================================================
    # TREE REFS
    # ============================================================
    def _add_tree_refs(self, elements: list, sections: list) -> list:
        element_to_section = {sec["element_id"]: sec["id"] for sec in sections}
        current_section_id = None

        for el in elements:
            if el["type"] == "section_header":
                current_section_id = element_to_section.get(el["id"])
                el["parent_id"] = None
            else:
                el["parent_id"] = current_section_id

            el["children_ids"] = []

        return elements

    # ============================================================
    # FIGURES / TABLES / PAGES
    # ============================================================
    def _extract_figures(self, elements: list) -> list:
        return [el for el in elements if el["type"] == "picture"]

    def _extract_tables(self, elements: list) -> list:
        return [el for el in elements if el["type"] == "table"]

    def _extract_pages(self, elements: list) -> list:
        pages: dict[int, dict] = {}
        for el in elements:
            page = el["page"]
            if page not in pages:
                pages[page] = {
                    "page_number": page,
                    "element_ids": [],
                    "section_ids": [],
                }
            pages[page]["element_ids"].append(el["id"])

            # section_ids
            if el.get("parent_id"):
                if el["parent_id"] not in pages[page]["section_ids"]:
                    pages[page]["section_ids"].append(el["parent_id"])

        return list(pages.values())

    # ============================================================
    # SPLIT
    # ============================================================
    def _split_broken_elements(self) -> list:
        result = []
        for el in self.elements:
            splits = self._try_split(el)
            if splits is None:
                result.append(el)
            else:
                result.extend(splits)
        return result

    def _try_split(self, el: dict) -> Optional[list]:
        text = el["text"]
        parent = el.get("parent_section")

        # الگو ۱: "} Example : ... 3. while Loop :"
        m = re.match(
            r"^([}\]]+)\s+(Example|Syntax)\s*:\s*(.+?)\s+(\d+\.\s+[^:]{2,100}:)\s*$",
            text, re.DOTALL,
        )
        if m:
            closing, label, code_body, next_heading = m.groups()
            return [
                {**el, "type": "code", "text": closing.strip(), "level": None},
                {**el, "type": "section_header", "text": f"{label} :",
                 "level": 3, "parent_section": parent},
                {**el, "type": "code", "text": code_body.strip(), "level": None,
                 "parent_section": parent},
                {**el, "type": "section_header", "text": next_heading.strip(),
                 "level": 2, "parent_section": parent},
            ]

        # الگو ۲: "... } 3. while Loop :"
        m = re.match(r"^(.+?)\s+(\d+\.\s+[^:]{2,100}:)\s*$", text, re.DOTALL)
        if m and el["type"] in ("code", "paragraph"):
            body, next_heading = m.groups()
            return [
                {**el, "text": body.strip()},
                {**el, "type": "section_header", "text": next_heading.strip(),
                 "level": 2, "parent_section": parent},
            ]

        # الگو ۳: "Example : $value = ..."
        m = re.match(r"^(Example|Syntax)\s*:\s+(.+)$", text, re.DOTALL)
        if m and el["type"] == "code":
            label, body = m.groups()
            return [
                {**el, "type": "section_header", "text": f"{label} :",
                 "level": 3, "parent_section": parent},
                {**el, "type": "code", "text": body.strip(), "level": None},
            ]

        return None

    # ============================================================
    # MERGE
    # ============================================================
    def _merge_fragments(self, elements: list) -> list:
        merged = []
        i = 0
        while i < len(elements):
            el = elements[i]
            if self._is_incomplete_fragment(el) and i + 1 < len(elements):
                next_el = elements[i + 1]
                if next_el["type"] in ("code", "paragraph", "list_item"):
                    merged_text = f"{el['text']} {next_el['text']}"
                    merged.append({**el, "text": merged_text, "type": "paragraph"})
                    i += 2
                    continue
            merged.append(el)
            i += 1
        return merged

    def _is_incomplete_fragment(self, el: dict) -> bool:
        text = el["text"].strip()
        if len(text) > 20:
            return False
        if not text or not text[0].isupper():
            return False
        if text.endswith((".", ":", ";", "}", ")")):
            return False
        return True

    # ============================================================
    # HELPERS
    # ============================================================
    def _make_id(self, page: int, order: int) -> str:
        return f"doc{self.document_id}_p{page}_e{order}"

    def _get_title(self, elements: list) -> str:
        for el in elements:
            if el["type"] == "section_header":
                return el["text"]
        return "Untitled"

    def _get_stats(self, elements: list) -> dict:
        stats = {}
        for el in elements:
            stats[el["type"]] = stats.get(el["type"], 0) + 1
        return stats