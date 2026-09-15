import re
from docling_core.types.doc import DoclingDocument


class DoclingInspector:
    """
    تبدیل DoclingDocument به ساختار غنی و سلسله‌مراتبی.

    خروجی هر element:
      - order: شماره ترتیبی
      - type: section_header | paragraph | list_item | code | table | picture
      - text: متن کامل
      - page: شماره صفحه
      - level: 1 | 2 | 3 | None
      - parent_section: عنوان heading والد
    """

    # ============================================================
    # CONFIG
    # ============================================================
    TYPE_MAP = {
        "text": "paragraph",
        "section_header": "section_header",
        "list_item": "list_item",
        "code": "code",
        "caption": "caption",
        "formula": "formula",
        "table": "table",
        "picture": "picture",
    }

    HEADING_LABELS = {"section_header"}
    IGNORE_LABELS = {"page_header", "page_footer"}

    HEADING_KEYWORDS = [
        "operator", "statement", "loop", "block", "function",
        "class", "method", "autoloading", "path", "style",
        "example", "syntax", "include", "require", "assignment",
        "arithmetic", "comparison", "bitwise", "logical", "casting",
    ]

    # ============================================================
    # INIT
    # ============================================================
    def __init__(self, doc: DoclingDocument):
        self.doc = doc
        self._elements = None

    # ============================================================
    # PUBLIC API
    # ============================================================
    def inspect(self) -> dict:
        elements = self.get_elements()
        return {
            "title": self.get_title(),
            "elements": elements,
            "stats": self.get_stats(elements),
            "tree": self.get_tree(elements),
        }

    # ============================================================
    # GET ELEMENTS (قلب پروژه)
    # ============================================================
    def get_elements(self) -> list:
        if self._elements is not None:
            return self._elements

        elements = []
        order = 0
        section_stack: list[tuple[int, str]] = []

        # ---------- TEXT ELEMENTS ----------
        for item in self.doc.texts:
            if item.label in self.IGNORE_LABELS:
                continue

            text = self._clean_text(item.text)
            if not text:
                continue

            order += 1

            is_heading = (
                item.label in self.HEADING_LABELS
                or self._looks_like_heading(text, item.label)
            )

            if is_heading:
                level = self._infer_level(text)

                # Syntax/Example: به استک اضافه نمی‌شن
                is_label_heading = bool(
                    re.match(r"^(Syntax|Example)\s*:", text, re.IGNORECASE)
                )

                if is_label_heading:
                    parent_section = (
                        section_stack[-1][1] if section_stack else None
                    )
                else:
                    while section_stack and section_stack[-1][0] >= level:
                        section_stack.pop()

                    parent_section = (
                        section_stack[-1][1] if section_stack else None
                    )
                    section_stack.append((level, text))

                elements.append({
                    "order": order,
                    "type": "section_header",
                    "text": text,
                    "page": self._get_page_number(item),
                    "level": level,
                    "parent_section": parent_section,
                })
            else:
                parent_section = (
                    section_stack[-1][1] if section_stack else None
                )

                elements.append({
                    "order": order,
                    "type": self._normalize_type(item.label),
                    "text": text,
                    "page": self._get_page_number(item),
                    "level": None,
                    "parent_section": parent_section,
                })

        # ---------- TABLES ----------
        for table in self.doc.tables:
            order += 1
            elements.append({
                "order": order,
                "type": "table",
                "text": table.export_to_markdown() if hasattr(table, "export_to_markdown") else "",
                "page": self._get_page_number(table),
                "level": None,
                "parent_section": section_stack[-1][1] if section_stack else None,
            })

        # ---------- PICTURES ----------
        for pic in self.doc.pictures:
            order += 1
            elements.append({
                "order": order,
                "type": "picture",
                "text": self._get_caption(pic) or "",
                "page": self._get_page_number(pic),
                "level": None,
                "parent_section": section_stack[-1][1] if section_stack else None,
            })

        # ---------- SORT ----------
        elements.sort(key=lambda e: (e["page"] or 0, e["order"]))

        for i, el in enumerate(elements, start=1):
            el["order"] = i

        self._elements = elements
        return elements

    # ============================================================
    # TREE (درخت سلسله‌مراتبی)
    # ============================================================
    def get_tree(self, elements: list) -> dict:
        tree = {"root": {"children": []}}
        stack = [tree["root"]]

        for el in elements:
            if el["type"] == "section_header":
                level = el["level"] or 1

                while len(stack) > level:
                    stack.pop()

                node = {
                    "text": el["text"],
                    "page": el["page"],
                    "level": level,
                    "order": el["order"],
                    "children": [],
                    "items": [],
                }
                stack[-1]["children"].append(node)
                stack.append(node)
            else:
                if stack:
                    stack[-1].setdefault("items", []).append({
                        "type": el["type"],
                        "page": el["page"],
                        "order": el["order"],
                        "text": el["text"][:120],
                    })

        return tree

    # ============================================================
    # TITLE / STATS
    # ============================================================
    def get_title(self) -> str:
        for item in self.doc.texts:
            if item.label in self.HEADING_LABELS:
                text = self._clean_text(item.text)
                if text:
                    return text
        return "Untitled"

    def get_stats(self, elements: list) -> dict:
        stats = {}
        for el in elements:
            stats[el["type"]] = stats.get(el["type"], 0) + 1
        return stats

    # ============================================================
    # PRETTY PRINT
    # ============================================================
    def pretty_print(self, max_text_len: int = 200) -> str:
        data = self.inspect()
        lines = []

        lines.append("=" * 60)
        lines.append("DOCUMENT")
        lines.append("=" * 60)
        lines.append("")
        lines.append("TITLE:")
        lines.append(data["title"])
        lines.append("")
        lines.append("STATS:")
        for k, v in sorted(data["stats"].items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("ELEMENTS")
        lines.append("-" * 60)
        lines.append("")

        for el in data["elements"]:
            lines.append(f"[{el['order']}]")
            lines.append(f"type: {el['type']}")
            lines.append(f"page: {el['page']}")
            if el.get("level"):
                lines.append(f"level: {el['level']}")
            if el.get("parent_section"):
                lines.append(f"parent_section: {el['parent_section']}")

            text = el["text"]
            if len(text) > max_text_len:
                text = text[:max_text_len] + "..."
            lines.append(f"text: {text}")
            lines.append("")

        return "\n".join(lines)

    # ============================================================
    # HELPERS
    # ============================================================
    def _normalize_type(self, label: str) -> str:
        return self.TYPE_MAP.get(label, label)

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _get_page_number(self, item):
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            return prov[0].page_no
        return None

    def _get_caption(self, item) -> str:
        captions = getattr(item, "captions", None)
        if not captions:
            return ""
        texts = []
        for ref in captions:
            ref_str = getattr(ref, "self_ref", None) or str(ref)
            for t in self.doc.texts:
                if getattr(t, "self_ref", None) == ref_str:
                    texts.append(self._clean_text(t.text))
        return " ".join(texts)

    # ============================================================
    # DETECTORS
    # ============================================================
    def _looks_like_heading(self, text: str, label: str) -> bool:
        """تشخیص heading هایی که docling به عنوان list_item دیده."""
        if label != "list_item":
            return False

        text = text.strip()

        if len(text) > 100:
            return False

        if not text.endswith(":"):
            return False

        # 1) "X. Title (op) :"
        if re.match(r"^\d+\.\s+.+\(.+\)\s*:$", text):
            return True

        # 2) "Title (op) :"
        if re.search(r"\([^)]+\)\s*:$", text) and len(text) < 80:
            return True

        # 3) "Title Words :" (کوتاه + کلمات کلیدی)
        if len(text) < 60 and re.match(r"^[A-Za-z][A-Za-z\s,._\-()/+=]*:$", text):
            text_lower = text.lower()
            if any(kw in text_lower for kw in self.HEADING_KEYWORDS):
                return True

        return False

    def _infer_level(self, text: str) -> int:
        """استنتاج سطح heading از روی الگوی متن."""
        text = text.strip()

        # level 3: Syntax / Example
        if re.match(r"^(Syntax|Example)\s*:", text, re.IGNORECASE):
            return 3

        # level 1: شروع با نماد
        if text.startswith(("➢", "·", "▪", "➤", "▶", "●")):
            return 1

        # level 2: "Instanceof Operator :" و "Type Operators :"
        if re.match(r"^[A-Z][a-zA-Z\s]*(Operator|Operators)\s*(\([^)]*\))?\s*:$", text):
            return 2

        # level 2: شماره + پرانتز
        if re.match(r"^\d+\.\s+", text) and re.search(r"\([^)]*\)", text[:60]):
            return 2

        # level 1: شماره + حرف بزرگ + بدون پرانتز
        if re.match(r"^\d+\.\s+[A-Z]", text) and not re.search(r"[()\[\]]", text[:60]):
            return 1

        # level 2: شماره‌دار دیگه
        if re.match(r"^\d+\.\s+", text):
            return 2

        # level 2: عنوان با پرانتز در انتها
        if re.search(r"\([^)]+\)\s*:$", text):
            return 2

        return 1