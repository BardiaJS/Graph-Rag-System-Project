import re

from typing import Optional


class DoclingNormalizer:
    """
    عناصر خام DoclingInspector را به NormalizedDocument تبدیل می‌کند.

    خروجی:

      - elements: لیست تخت عناصر با:
          section_path
          parent_section
          parent_id
          children_ids
          order
          id

      - sections: درخت سلسله‌مراتبی section ها

      - figures: لیست تصاویر

      - tables: لیست جدول‌ها

      - pages: لیست صفحات با element_ids و section_ids
    """

    LABEL_PATTERN = re.compile(
        r"^(Syntax|Example)\s*:$",
        re.IGNORECASE,
    )

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

        # --------------------------------------------------------
        # 1. Split broken elements
        # --------------------------------------------------------

        split_elements = self._split_broken_elements()

        # --------------------------------------------------------
        # 2. Merge fragments
        # --------------------------------------------------------

        merged_elements = self._merge_fragments(split_elements)

        # --------------------------------------------------------
        # 3. Recompute section paths
        # --------------------------------------------------------

        with_paths = self._recompute_section_paths(
            merged_elements
        )

        # --------------------------------------------------------
        # 4. Assign order + stable IDs
        # --------------------------------------------------------

        for i, el in enumerate(with_paths, start=1):

            el["order"] = i

            el["id"] = self._make_id(
                el["page"],
                i,
            )

        # --------------------------------------------------------
        # 5. Build hierarchical section tree
        # --------------------------------------------------------

        sections = self._build_sections_tree(
            with_paths
        )

        # --------------------------------------------------------
        # 6. Add parent / children references
        # --------------------------------------------------------

        with_refs = self._add_tree_refs(
            with_paths,
            sections,
        )

        # --------------------------------------------------------
        # 7. Extract figures / tables / pages
        # --------------------------------------------------------

        figures = self._extract_figures(
            with_refs
        )

        tables = self._extract_tables(
            with_refs
        )

        pages = self._extract_pages(
            with_refs
        )

        # --------------------------------------------------------
        # 8. Final normalized document
        # --------------------------------------------------------

        self._normalized = {
            "document_id": self.document_id,

            "title": self._get_title(
                with_refs
            ),

            "stats": self._get_stats(
                with_refs
            ),

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

    def _recompute_section_paths(
        self,
        elements: list,
    ) -> list:

        """
        ساخت hierarchy بر اساس level مربوط به section header.

        مثال:

            7. Logical Operator
                level = 1

                1. Logical AND
                    level = 2

                Example:
                    section_path همان parent sectionها است.

        توجه:

            Example:
            Syntax:

        section واقعی نیستند و فقط label داخل section هستند.
        """

        stack: list[tuple[int, str]] = []

        for el in elements:

            text = str(
                el.get("text", "")
            ).strip()

            # ----------------------------------------------------
            # SECTION HEADER
            # ----------------------------------------------------

            if el["type"] == "section_header":

                # ----------------------------------------------
                # Example / Syntax
                # ----------------------------------------------

                if self._is_label(text):

                    el["section_path"] = [
                        section_title
                        for _, section_title in stack
                    ]

                    el["section_level"] = (
                        len(stack)
                    )

                    el["parent_section"] = (
                        stack[-1][1]
                        if stack
                        else None
                    )

                    continue

                # ----------------------------------------------
                # Real section
                # ----------------------------------------------

                level = el.get("level") or 1

                level = int(level)

                # Remove sections at same or deeper level
                while (
                    stack
                    and stack[-1][0] >= level
                ):
                    stack.pop()

                parent = (
                    stack[-1][1]
                    if stack
                    else None
                )

                stack.append(
                    (
                        level,
                        text,
                    )
                )

                el["section_path"] = [
                    section_title
                    for _, section_title in stack
                ]

                el["section_level"] = level

                el["parent_section"] = parent

                continue

            # ----------------------------------------------------
            # NORMAL ELEMENT
            # ----------------------------------------------------

            el["section_path"] = [
                section_title
                for _, section_title in stack
            ]

            el["section_level"] = (
                len(stack)
            )

            el["parent_section"] = (
                stack[-1][1]
                if stack
                else None
            )

        return elements

    # ============================================================
    # SECTIONS TREE
    # ============================================================

    def _build_sections_tree(
        self,
        elements: list,
    ) -> list:

        """
        ساخت درخت واقعی sectionها.

        Example / Syntax section محسوب نمی‌شوند.

        ساختار:

            Section A
            ├── Section B
            │   ├── paragraph
            │   └── code
            │
            └── Section C
        """

        sections = []

        # stack:
        #
        # [
        #     (section_dict, level),
        #     ...
        # ]

        stack = []

        for el in elements:

            # ----------------------------------------------------
            # Normal element
            # ----------------------------------------------------

            if el["type"] != "section_header":

                if stack:

                    stack[-1][0][
                        "element_ids"
                    ].append(
                        el["id"]
                    )

                continue

            text = str(
                el.get("text", "")
            ).strip()

            # ----------------------------------------------------
            # Example / Syntax
            # ----------------------------------------------------

            if self._is_label(text):

                if stack:

                    stack[-1][0][
                        "element_ids"
                    ].append(
                        el["id"]
                    )

                continue

            # ----------------------------------------------------
            # Real section
            # ----------------------------------------------------

            level = int(
                el.get("level") or 1
            )

            section = {
                "id": (
                    f"section_{el['id']}"
                ),

                "title": text,

                "level": level,

                "page": el["page"],

                "element_id": el["id"],

                "parent_section_id": None,

                "children_section_ids": [],

                "element_ids": [],

                "section_path": (
                    el["section_path"]
                ),
            }

            # ----------------------------------------------------
            # Find parent section
            # ----------------------------------------------------

            while (
                stack
                and stack[-1][1] >= level
            ):
                stack.pop()

            if stack:

                parent = stack[-1][0]

                parent[
                    "children_section_ids"
                ].append(
                    section["id"]
                )

                section[
                    "parent_section_id"
                ] = parent["id"]

            # ----------------------------------------------------
            # Store section
            # ----------------------------------------------------

            sections.append(
                section
            )

            stack.append(
                (
                    section,
                    level,
                )
            )

        return sections

    # ============================================================
    # TREE REFS
    # ============================================================

    def _add_tree_refs(
        self,
        elements: list,
        sections: list,
    ) -> list:

        """
        parent_id را برای عناصر مشخص می‌کند.

        مثال:

            Section A
            parent_id = None

            Section B
            parent_id = section_A_id

            paragraph داخل B
            parent_id = section_B_id

            Example:
            parent_id = section_B_id
        """

        # --------------------------------------------------------
        # element_id -> section object
        # --------------------------------------------------------

        element_to_section = {
            sec["element_id"]: sec
            for sec in sections
        }

        current_section = None

        for el in elements:

            text = str(
                el.get("text", "")
            ).strip()

            # ====================================================
            # SECTION HEADER
            # ====================================================

            if el["type"] == "section_header":

                # ------------------------------------------------
                # Example / Syntax
                # ------------------------------------------------

                if self._is_label(text):

                    el["parent_id"] = (
                        current_section["id"]
                        if current_section
                        else None
                    )

                # ------------------------------------------------
                # Real section
                # ------------------------------------------------

                else:

                    section = (
                        element_to_section.get(
                            el["id"]
                        )
                    )

                    if section is None:

                        el["parent_id"] = None

                        current_section = None

                    else:

                        # Parent section
                        el["parent_id"] = (
                            section[
                                "parent_section_id"
                            ]
                        )

                        # Current section
                        current_section = (
                            section
                        )

            # ====================================================
            # NORMAL ELEMENT
            # ====================================================

            else:

                el["parent_id"] = (
                    current_section["id"]
                    if current_section
                    else None
                )

            # ----------------------------------------------------
            # Children
            # ----------------------------------------------------

            el["children_ids"] = []

        return elements

    # ============================================================
    # FIGURES / TABLES / PAGES
    # ============================================================

    def _extract_figures(
        self,
        elements: list,
    ) -> list:

        return [
            el
            for el in elements
            if el["type"] == "picture"
        ]

    def _extract_tables(
        self,
        elements: list,
    ) -> list:

        return [
            el
            for el in elements
            if el["type"] == "table"
        ]

    def _extract_pages(
        self,
        elements: list,
    ) -> list:

        pages: dict[int, dict] = {}

        for el in elements:

            page = el["page"]

            if page not in pages:

                pages[page] = {
                    "page_number": page,
                    "element_ids": [],
                    "section_ids": [],
                }

            # ----------------------------------------------------
            # Element
            # ----------------------------------------------------

            pages[page][
                "element_ids"
            ].append(
                el["id"]
            )

            # ----------------------------------------------------
            # Section
            # ----------------------------------------------------

            parent_id = el.get(
                "parent_id"
            )

            if parent_id:

                if (
                    parent_id
                    not in pages[page][
                        "section_ids"
                    ]
                ):

                    pages[page][
                        "section_ids"
                    ].append(
                        parent_id
                    )

        return list(
            pages.values()
        )

    # ============================================================
    # SPLIT
    # ============================================================

    def _split_broken_elements(
        self,
    ) -> list:

        result = []

        for el in self.elements:

            splits = self._try_split(
                el
            )

            if splits is None:

                result.append(el)

            else:

                result.extend(
                    splits
                )

        return result

    def _try_split(
        self,
        el: dict,
    ) -> Optional[list]:

        text = str(
            el.get("text", "")
        )

        parent = el.get(
            "parent_section"
        )

        # ========================================================
        # Pattern 1
        #
        # "} Example : code 3. while Loop :"
        # ========================================================

        m = re.match(
            r"^([}\]])+\s+"
            r"(Example|Syntax)\s*:\s*"
            r"(.+?)\s+"
            r"(\d+\.\s+[^:]{2,100}:)\s*$",
            text,
            re.DOTALL,
        )

        if m:

            closing, label, code_body, next_heading = (
                m.groups()
            )

            return [

                {
                    **el,
                    "type": "code",
                    "text": closing.strip(),
                    "level": None,
                },

                {
                    **el,
                    "type": "section_header",
                    "text": f"{label} :",
                    "level": 3,
                    "parent_section": parent,
                },

                {
                    **el,
                    "type": "code",
                    "text": code_body.strip(),
                    "level": None,
                    "parent_section": parent,
                },

                {
                    **el,
                    "type": "section_header",
                    "text": next_heading.strip(),
                    "level": 2,
                    "parent_section": parent,
                },
            ]

        # ========================================================
        # Pattern 2
        #
        # "... } 3. while Loop :"
        # ========================================================

        m = re.match(
            r"^(.+?)\s+"
            r"(\d+\.\s+[^:]{2,100}:)\s*$",
            text,
            re.DOTALL,
        )

        if (
            m
            and el["type"]
            in ("code", "paragraph")
        ):

            body, next_heading = (
                m.groups()
            )

            return [

                {
                    **el,
                    "text": body.strip(),
                },

                {
                    **el,
                    "type": "section_header",
                    "text": next_heading.strip(),
                    "level": 2,
                    "parent_section": parent,
                },
            ]

        # ========================================================
        # Pattern 3
        #
        # "Example : $value = ..."
        # ========================================================

        m = re.match(
            r"^(Example|Syntax)\s*:\s+(.+)$",
            text,
            re.DOTALL,
        )

        if (
            m
            and el["type"] == "code"
        ):

            label, body = m.groups()

            return [

                {
                    **el,
                    "type": "section_header",
                    "text": f"{label} :",
                    "level": 3,
                    "parent_section": parent,
                },

                {
                    **el,
                    "type": "code",
                    "text": body.strip(),
                    "level": None,
                },
            ]

        return None

    # ============================================================
    # MERGE
    # ============================================================

    def _merge_fragments(
        self,
        elements: list,
    ) -> list:

        """
        Fragmentهای ناقص را با عنصر بعدی merge می‌کند.

        اما:

            section_header

        هرگز merge نمی‌شود.
        """

        merged = []

        i = 0

        while i < len(elements):

            el = elements[i]

            # ----------------------------------------------------
            # Never merge section headers
            # ----------------------------------------------------

            if el["type"] == "section_header":

                merged.append(el)

                i += 1

                continue

            # ----------------------------------------------------
            # Try merge
            # ----------------------------------------------------

            if (
                self._is_incomplete_fragment(el)
                and i + 1 < len(elements)
            ):

                next_el = elements[
                    i + 1
                ]

                if next_el["type"] in (
                    "code",
                    "paragraph",
                    "list_item",
                ):

                    merged_text = (
                        f"{el['text']} "
                        f"{next_el['text']}"
                    )

                    merged.append(
                        {
                            **el,
                            "text": merged_text,
                            "type": "paragraph",
                        }
                    )

                    i += 2

                    continue

            merged.append(el)

            i += 1

        return merged

    def _is_incomplete_fragment(
        self,
        el: dict,
    ) -> bool:

        # --------------------------------------------------------
        # Never treat section headers as fragments
        # --------------------------------------------------------

        if el["type"] == "section_header":
            return False

        text = str(
            el.get("text", "")
        ).strip()

        # Empty
        if not text:
            return False

        # Long text is probably complete
        if len(text) > 20:
            return False

        # Must start with uppercase
        if not text[0].isupper():
            return False

        # Already terminated
        if text.endswith(
            (
                ".",
                ":",
                ";",
                "}",
                ")",
            )
        ):
            return False

        return True

    # ============================================================
    # HELPERS
    # ============================================================

    def _is_label(
        self,
        text: str,
    ) -> bool:

        """
        آیا متن فقط یک label از نوع:

            Example:
            Syntax:

        است؟
        """

        return bool(
            self.LABEL_PATTERN.match(
                text.strip()
            )
        )

    def _make_id(
        self,
        page: int,
        order: int,
    ) -> str:

        return (
            f"doc{self.document_id}"
            f"_p{page}"
            f"_e{order}"
        )

    def _get_title(
        self,
        elements: list,
    ) -> str:

        for el in elements:

            if (
                el["type"]
                == "section_header"
            ):

                text = str(
                    el.get("text", "")
                ).strip()

                if not self._is_label(
                    text
                ):
                    return text

        return "Untitled"

    def _get_stats(
        self,
        elements: list,
    ) -> dict:

        stats = {}

        for el in elements:

            element_type = el[
                "type"
            ]

            stats[element_type] = (
                stats.get(
                    element_type,
                    0,
                )
                + 1
            )

        return stats
