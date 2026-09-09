from statistics import mean, median
import re

class ChunkEvaluator:

    def __init__(self, chunks):
        self.chunks = chunks

    def evaluate(self):

        if not self.chunks:
            return {
                "total_chunks": 0,
                "message": "No chunks found"
            }

        lengths = [
            len(chunk.get("text", ""))
            for chunk in self.chunks
        ]

        empty_chunks = sum(
            1
            for chunk in self.chunks
            if not chunk.get("text", "").strip()
        )

        very_short_chunks = sum(
            1
            for length in lengths
            if 100 <= length < 300
        )

        very_long_chunks = sum(
            1
            for length in lengths
            if length > 800
        )

        missing_metadata = sum(
            1
            for chunk in self.chunks
            if (
                chunk.get("document_id") is None
                or chunk.get("page_number") is None
                or chunk.get("chunk_index") is None
            )
        )

        return {
            "total_chunks": len(self.chunks),

            "min_length": min(lengths),
            "max_length": max(lengths),
            "mean_length": round(mean(lengths), 2),
            "median_length": median(lengths),

            "empty_chunks": empty_chunks,
            "very_short_chunks": very_short_chunks,
            "very_long_chunks": very_long_chunks,

            "missing_metadata": missing_metadata,
        }


    def evaluate_boundaries(self):
        results = []

        for i in range(len(self.chunks) - 1):

            current = self.chunks[i]
            next_chunk = self.chunks[i + 1]

            current_text = current["text"].strip()
            next_text = next_chunk["text"].strip()

            current_page = current.get("page_number")
            next_page = next_chunk.get("page_number")

            overlap = self._calculate_overlap(
                current_text,
                next_text
            )

            result = {
                "current_chunk": current["chunk_index"],
                "next_chunk": next_chunk["chunk_index"],
                "current_page": current_page,
                "next_page": next_page,
                "current_end": current_text[-100:],
                "next_start": next_text[:100],
                "overlap": overlap,
                "category": "CLEAN",
            }

            # 1. تغییر صفحه
            if current_page != next_page:
                result["category"] = "PAGE_TRANSITION"

            # 2. شکستن واقعی کلمه
            elif self._is_word_split(current_text, next_text):
                result["category"] = "WORD_SPLIT"

            # 3. ادامه‌ی طبیعی به خاطر overlap
            elif overlap > 0 and self._is_sentence_continuation(
                current_text,
                next_text
            ):
                result["category"] = "OVERLAP_CONTINUATION"

            # 4. overlap داریم ولی پایان جمله تمیز است
            elif overlap > 0:
                result["category"] = "OVERLAP_CLEAN"

            # 5. بدون overlap ولی ظاهراً جمله ادامه پیدا کرده
            elif self._is_sentence_continuation(
                current_text,
                next_text
            ):
                result["category"] = "SUSPICIOUS"

            results.append(result)

        return results

    def _normalize(self, text):
        return re.sub(
            r"\s+",
            " ",
            text.strip()
        )

    def _ends_sentence(self, text):
        return bool(
            re.search(
                r"[.!?][\"')\]]*$",
                text
            )
        )

    def _is_word_split(self, current_text, next_text):
        if not current_text or not next_text:
            return False

        hyphen_chars = {
            "-",
            "‐",
            "-",
            "‒",
            "–",
            "—",
            "\u00ad",
        }

        last_char = current_text[-1]

        # مثال:
        # hash-
        # ing method
        if last_char in hyphen_chars:
            return True

        return False

    def _is_sentence_continuation(
        self,
        current,
        next_text
    ):

        if not current or not next_text:
            return False

        if self._ends_sentence(current):
            return False

        first_char = next_text[0]

        # اگر chunk بعدی با حرف کوچک شروع شود
        # و chunk قبلی جمله را تمام نکرده باشد
        if first_char.islower():
            return True

        # اگر chunk قبلی با , ; : تمام شده باشد
        # احتمال ادامه‌ی جمله زیاد است.
        if current[-1] in {",", ";", ":"}:
            return True

        return False

    def _calculate_overlap(
        self,
        current,
        next_text
    ):
        """
        طول بیشترین suffix از chunk فعلی
        که عیناً prefix از chunk بعدی است.
        """

        max_length = min(
            150,
            len(current),
            len(next_text)
        )

        for length in range(
            max_length,
            9,
            -1
        ):
            if current[-length:] == next_text[:length]:
                return length

        return 0



    def summarize_boundaries(self, boundaries):
        summary = {
            "total_boundaries": len(boundaries),
            "clean_boundaries": 0,
            "overlap_continuations": 0,
            "overlap_clean": 0,
            "page_transitions": 0,
            "word_splits": 0,
            "suspicious_boundaries": 0,
        }

        for boundary in boundaries:

            category = boundary["category"]

            if category == "CLEAN":
                summary["clean_boundaries"] += 1

            elif category == "OVERLAP_CONTINUATION":
                summary["overlap_continuations"] += 1

            elif category == "OVERLAP_CLEAN":
                summary["overlap_clean"] += 1

            elif category == "PAGE_TRANSITION":
                summary["page_transitions"] += 1

            elif category == "WORD_SPLIT":
                summary["word_splits"] += 1

            elif category == "SUSPICIOUS":
                summary["suspicious_boundaries"] += 1

        return summary