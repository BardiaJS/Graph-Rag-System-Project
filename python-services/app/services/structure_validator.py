class StructureValidator:
    """
    اعتبارسنجی ساختار NormalizedDocument.
    """

    def __init__(self, normalized_doc: dict):
        self.doc = normalized_doc
        self.errors = []
        self.warnings = []

    def validate(self) -> dict:
        self._check_orphan_elements()
        self._check_orphan_sections()
        self._check_section_path_consistency()
        self._check_duplicate_ids()
        self._check_empty_texts()
        self._check_pages_coverage()

        return {
            "is_valid": len(self.errors) == 0,
            "num_errors": len(self.errors),
            "num_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def _check_orphan_elements(self):
        """هر element غیر heading باید parent_id داشته باشه."""
        for el in self.doc["elements"]:
            if el["type"] != "section_header":
                if not el.get("parent_id"):
                    self.errors.append({
                        "type": "orphan_element",
                        "element_id": el["id"],
                        "message": f"Element '{el['id']}' parent ندارد",
                    })

    def _check_orphan_sections(self):
        """هر section غیر level 1 باید parent داشته باشه."""
        for sec in self.doc["sections"]:
            if sec["level"] > 1 and not sec.get("parent_section_id"):
                self.warnings.append({
                    "type": "orphan_section",
                    "section_id": sec["id"],
                    "message": f"Section '{sec['title']}' (level {sec['level']}) والد ندارد",
                })

    def _check_section_path_consistency(self):
        """section_path باید با section_level هماهنگ باشه."""
        for el in self.doc["elements"]:
            if el["type"] == "section_header":
                if len(el.get("section_path", [])) != el.get("section_level", 0):
                    self.warnings.append({
                        "type": "section_path_mismatch",
                        "element_id": el["id"],
                        "message": f"section_path length != section_level",
                    })

    def _check_duplicate_ids(self):
        """هیچ id تکراری نباشه."""
        ids = [el["id"] for el in self.doc["elements"]]
        seen = set()
        for id_ in ids:
            if id_ in seen:
                self.errors.append({
                    "type": "duplicate_id",
                    "id": id_,
                })
            seen.add(id_)

    def _check_empty_texts(self):
        """هیچ متن خالی نباشه (به جز caption)."""
        for el in self.doc["elements"]:
            if el["type"] not in ("caption", "picture") and not el["text"].strip():
                self.warnings.append({
                    "type": "empty_text",
                    "element_id": el["id"],
                })

    def _check_pages_coverage(self):
        """همه صفحات باید توی pages باشن."""
        pages_in_elements = {el["page"] for el in self.doc["elements"] if el.get("page")}
        pages_in_metadata = {p["page_number"] for p in self.doc["pages"]}
        missing = pages_in_elements - pages_in_metadata
        for page in missing:
            self.warnings.append({
                "type": "missing_page",
                "page": page,
            })