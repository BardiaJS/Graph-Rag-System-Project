import json
from pathlib import Path

import pymupdf4llm


class DocumentStructureExtractor:
    def __init__(self, file_path, document_id):
        self.file_path = Path(file_path)
        self.document_id = document_id

    def extract(self):
        raw_json = pymupdf4llm.to_json(str(self.file_path))
        data = json.loads(raw_json)

        return self._normalize(data)

    def _normalize(self, data):
        blocks = []

        for page in data.get("pages", []):
            page_number = page.get("page_number")

            for box in page.get("boxes", []):
                boxclass = box.get("boxclass")

                # حذف header و footer صفحات
                if boxclass in {"page-header", "page-footer"}:
                    continue

                block = {
                    "document_id": self.document_id,
                    "page_number": page_number,
                    "content_type": boxclass,
                    "text": self._extract_text(box),
                    "bbox": [
                        box.get("x0"),
                        box.get("y0"),
                        box.get("x1"),
                        box.get("y1"),
                    ],
                    "header_level": box.get("header_level", 0),
                }

                # تصویر
                if boxclass == "picture":
                    block["image"] = box.get("image")

                # جدول
                if boxclass == "table":
                    block["table"] = box.get("table")

                blocks.append(block)

        return {
            "document_id": self.document_id,
            "page_count": data.get("page_count", 0),
            "blocks": blocks,
        }

    def _extract_text(self, box):
        # بعضی boxها textlines ندارند یا مقدار آن‌ها None است
        textlines = box.get("textlines") or []

        lines = []

        for line in textlines:
            spans = line.get("spans") or []

            line_text = "".join(
                span.get("text", "")
                for span in spans
            )

            if line_text.strip():
                lines.append(line_text.strip())

        return "\n".join(lines)