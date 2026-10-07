import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ← حالا بقیه importها
LARAVEL_STORAGE_PATH = os.getenv(
    "LARAVEL_STORAGE_PATH",
    "/home/bardia/Desktop/graph-rag/backend/storage/app/public"
)

from typing import Union
import redis
from rq import Queue, Worker
from app.services.docling_service import DoclingService
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import redis
import os
import logging
from collections import Counter, defaultdict
from dotenv import load_dotenv

from app.document_processor import DocumentProcessor   # ← این خط اضافه شد

from pydantic import BaseModel
from typing import List, Optional

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(title="Graph RAG - PDF Processor")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Redis
# ============================================================

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)

try:
    redis_client.ping()
    logger.info("✅ Connected to Redis successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")


# ============================================================
# Pydantic
# ============================================================

class ProcessDocumentRequest(BaseModel):
    user_id: int
    session_id: int
    file_path: str
    document_ids: List[int]


# ============================================================
# ChunkEvaluator
# ============================================================

class ChunkEvaluator:
    """
    ارزیابی و گزارش‌گیری از chunk ها.
    """

    @staticmethod
    def _get_tokens(chunk: Dict[str, Any]) -> int:
        return chunk.get("tokens") or chunk.get("token_count") or 0

    @staticmethod
    def _get_content(chunk: Dict[str, Any]) -> str:
        return chunk.get("content") or chunk.get("text") or ""

    @staticmethod
    def _is_atomic(chunk: Dict[str, Any]) -> bool:
        return bool(chunk.get("atomic") or chunk.get("is_atomic"))

    @staticmethod
    def _is_code(chunk: Dict[str, Any]) -> bool:
        return bool(chunk.get("code") or chunk.get("contains_code"))

    @staticmethod
    def _is_picture(chunk: Dict[str, Any]) -> bool:
        return bool(chunk.get("picture") or chunk.get("contains_picture"))

    @staticmethod
    def _is_table(chunk: Dict[str, Any]) -> bool:
        return bool(chunk.get("table") or chunk.get("contains_table"))

    @staticmethod
    def _get_section_path(chunk: Dict[str, Any]) -> List[str]:
        if chunk.get("section_path"):
            return chunk["section_path"]
        if isinstance(chunk.get("section"), dict):
            if chunk["section"].get("path"):
                return chunk["section"]["path"]
        if chunk.get("parent_section_path"):
            return chunk["parent_section_path"]
        return []

    @staticmethod
    def _get_heading(chunk: Dict[str, Any]) -> Optional[str]:
        if chunk.get("heading"):
            return chunk["heading"]
        if isinstance(chunk.get("section"), dict):
            return chunk["section"].get("heading")
        path = ChunkEvaluator._get_section_path(chunk)
        if path:
            return path[-1]
        return None

    @staticmethod
    def _get_element_types(chunk: Dict[str, Any]) -> List[str]:
        if chunk.get("element_types"):
            return chunk["element_types"]
        if isinstance(chunk.get("source"), dict):
            return chunk["source"].get("element_types") or []
        return []

    def classify(self, chunk: Dict[str, Any]) -> str:
        if self._is_atomic(chunk):
            if self._is_code(chunk):
                return "atomic_code"
            if self._is_picture(chunk):
                return "atomic_picture"
            if self._is_table(chunk):
                return "atomic_table"
            return "atomic_other"
        return "text"

    def build_report(
        self,
        chunks: List[Dict[str, Any]],
        under_token_threshold: int = 100,
        small_token_threshold: int = 50,
    ) -> Dict[str, Any]:

        total = len(chunks)
        token_counts = [self._get_tokens(c) for c in chunks]

        by_type: Counter = Counter()
        for chunk in chunks:
            by_type[self.classify(chunk)] += 1

        under_threshold: Dict[str, list] = defaultdict(list)
        for chunk in chunks:
            tokens = self._get_tokens(chunk)
            if tokens < under_token_threshold:
                under_threshold[self.classify(chunk)].append({
                    "chunk_id": chunk.get("chunk_id"),
                    "tokens": tokens,
                    "heading": self._get_heading(chunk),
                    "section_path": self._get_section_path(chunk),
                    "element_types": self._get_element_types(chunk),
                    "content_preview": self._get_content(chunk)[:120],
                })

        small_text_chunks: list = []
        for chunk in chunks:
            tokens = self._get_tokens(chunk)

            if self._is_atomic(chunk):
                continue

            if self._is_code(chunk) or self._is_picture(chunk) or self._is_table(chunk):
                continue

            if tokens >= small_token_threshold:
                continue

            small_text_chunks.append({
                "chunk_id": chunk.get("chunk_id"),
                "tokens": tokens,
                "heading": self._get_heading(chunk),
                "section_path": self._get_section_path(chunk),
                "element_types": self._get_element_types(chunk),
                "content_preview": self._get_content(chunk)[:150],
            })

        small_text_chunks.sort(key=lambda x: x["tokens"])

        buckets = [
            (0, 20), (20, 50), (50, 100), (100, 200),
            (200, 300), (300, 400), (400, 512), (512, 10000),
        ]
        histogram = []
        for lo, hi in buckets:
            count = sum(1 for t in token_counts if lo <= t < hi)
            histogram.append({
                "range": f"{lo}-{hi}",
                "count": count,
                "percent": round(count / max(total, 1) * 100, 1),
            })

        stats = {
            "total_chunks": total,
            "total_tokens": sum(token_counts),
            "avg_tokens": round(sum(token_counts) / max(total, 1), 1),
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "chunks_over_512": sum(1 for t in token_counts if t > 512),
            f"chunks_under_{under_token_threshold}": sum(
                1 for t in token_counts if t < under_token_threshold
            ),
            f"chunks_under_{small_token_threshold}": sum(
                1 for t in token_counts if t < small_token_threshold
            ),
            "chunks_under_20": sum(1 for t in token_counts if t < 20),
            "non_atomic_text_under_50": len(small_text_chunks),
        }

        return {
            "stats": stats,
            "by_type": dict(by_type),
            "histogram": histogram,
            "under_threshold": {
                k: {"count": len(v), "examples": v[:10]}
                for k, v in under_threshold.items()
            },
            "small_non_atomic_text_chunks": {
                "threshold": small_token_threshold,
                "count": len(small_text_chunks),
                "chunks": small_text_chunks,
            },
        }

    def print_report(
        self,
        report: Dict[str, Any],
        under_token_threshold: int = 100,
        small_token_threshold: int = 50,
    ):
        print("\n" + "=" * 80)
        print("CHUNK REPORT")
        print("=" * 80)

        stats = report["stats"]

        print("\n📊 STATS")
        print(f"  Total chunks                        : {stats['total_chunks']}")
        print(f"  Total tokens                        : {stats['total_tokens']}")
        print(f"  Avg tokens/chunk                    : {stats['avg_tokens']}")
        print(f"  Min tokens                          : {stats['min_tokens']}")
        print(f"  Max tokens                          : {stats['max_tokens']}")
        print(f"  Chunks > 512 tokens                 : {stats['chunks_over_512']}")
        print(f"  Chunks < {under_token_threshold} tokens                : {stats[f'chunks_under_{under_token_threshold}']}")
        print(f"  Chunks < {small_token_threshold} tokens                 : {stats[f'chunks_under_{small_token_threshold}']}")
        print(f"  Chunks < 20 tokens                  : {stats['chunks_under_20']}")
        print(f"  ✅ Non-atomic text chunks < {small_token_threshold}       : {stats['non_atomic_text_under_50']}")

        print("\n🏷️  BY TYPE")
        for type_name, count in sorted(report["by_type"].items()):
            print(f"  {type_name:20s}: {count}")

        print("\n📈 TOKEN DISTRIBUTION")
        for bucket in report["histogram"]:
            bar = "█" * int(bucket["percent"] / 2)
            print(
                f"  {bucket['range']:12s} : "
                f"{bucket['count']:4d} ({bucket['percent']:5.1f}%)  {bar}"
            )

        small = report["small_non_atomic_text_chunks"]

        print("\n" + "=" * 80)
        print(f"🔍 NON-ATOMIC TEXT CHUNKS < {small['threshold']} TOKENS")
        print("=" * 80)
        print(f"\n  Total: {small['count']} chunks\n")

        if small["count"] == 0:
            print("  ✅ هیچ chunk متنی کوچیکی وجود نداره!\n")
        else:
            for i, ch in enumerate(small["chunks"], start=1):
                print(f"  [{i}] chunk_id: {ch['chunk_id']}")
                print(f"      tokens  : {ch['tokens']}")
                print(f"      heading : {ch['heading']}")
                print(f"      path    : {ch['section_path']}")
                print(f"      types   : {ch['element_types']}")
                print(f"      preview : {ch['content_preview'][:100]}...")
                print()

        print("=" * 80)
        print(f"⚠️  CHUNKS UNDER {under_token_threshold} TOKENS (by type)")
        print("=" * 80)

        for type_name, data in sorted(report["under_threshold"].items()):
            print(f"\n  {type_name}: {data['count']} chunks")

        print("\n" + "=" * 80 + "\n")


# ============================================================
# ChunkMetadataCleaner
# ============================================================

class ChunkMetadataCleaner:
    """
    metadata chunk ها رو تمیز می‌کنه.
    """

    @staticmethod
    def _get_tokens(chunk: Dict[str, Any]) -> int:
        return chunk.get("tokens") or chunk.get("token_count") or 0

    @staticmethod
    def _get_content(chunk: Dict[str, Any]) -> str:
        return chunk.get("content") or chunk.get("text") or ""

    @staticmethod
    def _get_element_ids(chunk: Dict[str, Any]) -> List[str]:
        if chunk.get("element_ids"):
            return chunk["element_ids"]
        if isinstance(chunk.get("source"), dict):
            return chunk["source"].get("element_ids") or []
        return []

    @staticmethod
    def _get_element_types(chunk: Dict[str, Any]) -> List[str]:
        if chunk.get("element_types"):
            return chunk["element_types"]
        if isinstance(chunk.get("source"), dict):
            return chunk["source"].get("element_types") or []
        return []

    def clean(self, chunk: Dict[str, Any]) -> Dict[str, Any]:

        chunk_id = chunk.get("chunk_id")
        chunk_index = chunk.get("chunk_index")
        content = self._get_content(chunk)
        tokens = self._get_tokens(chunk)
        parent_chunk_id = chunk.get("parent_chunk_id")

        atomic = bool(chunk.get("atomic") or chunk.get("is_atomic"))
        code = bool(chunk.get("code") or chunk.get("contains_code"))
        picture = bool(chunk.get("picture") or chunk.get("contains_picture"))
        table = bool(chunk.get("table") or chunk.get("contains_table"))

        section_path = (
            chunk.get("section_path")
            or (chunk.get("section", {}) or {}).get("path")
            or chunk.get("parent_section_path")
            or []
        )
        section_level = (
            chunk.get("level")
            or chunk.get("section_level")
            or (chunk.get("section", {}) or {}).get("level")
            or 1
        )
        heading = (
            chunk.get("heading")
            or (chunk.get("section", {}) or {}).get("heading")
            or (section_path[-1] if section_path else None)
        )

        element_ids = self._get_element_ids(chunk)
        section_id = f"section_{element_ids[0]}" if element_ids else None

        element_types = self._get_element_types(chunk)
        page_start = (
            chunk.get("page_start")
            or (chunk.get("source", {}) or {}).get("page_start")
            or (min(chunk["pages"]) if chunk.get("pages") else None)
        )
        page_end = (
            chunk.get("page_end")
            or (chunk.get("source", {}) or {}).get("page_end")
            or (max(chunk["pages"]) if chunk.get("pages") else None)
        )

        return {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "content": content,
            "tokens": tokens,
            "atomic": atomic,
            "code": code,
            "picture": picture,
            "table": table,
            "parent_chunk_id": parent_chunk_id,
            "section": {
                "id": section_id,
                "path": section_path,
                "level": section_level,
                "heading": heading,
            },
            "source": {
                "element_ids": element_ids,
                "element_types": element_types,
                "page_start": page_start,
                "page_end": page_end,
            },
        }

    def clean_all(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.clean(c) for c in chunks]


# ============================================================
# ENDPOINT: /process
# ============================================================

from app.document_processor import DocumentProcessor
import asyncio
from concurrent.futures import ThreadPoolExecutor

# فقط ۱ پردازش همزمان — جلوگیری از OOM
executor = ThreadPoolExecutor(max_workers=1)


class ProcessRequest(BaseModel):
    user_id: Union[str, int]
    session_id: Union[str, int]
    file_path: str
    document_ids: List[Union[int, str]]
from app.services.embedding_service import upload_chunks


@app.post("/process")
async def process_document(request: ProcessRequest):
    try:
        full_path = os.path.join(LARAVEL_STORAGE_PATH, request.file_path)
        logger.info(f"Processing file: {full_path}")

        loop = asyncio.get_event_loop()

        def process():
            processor = DocumentProcessor(full_path)
            markdown = processor.docling_service()
            chunks = processor.chunking_service()

            # ← اینجا chunkها رو توی Qdrant ذخیره کن
            if chunks:
                count = upload_chunks(
                    chunks,
                    user_id=str(request.user_id),
                    document_id=int(request.document_ids[0]),
                )
                logger.info(f"Uploaded {count} chunks to Qdrant")

            return {"markdown": markdown, "chunks": chunks}

        result = await asyncio.wait_for(
            loop.run_in_executor(executor, process),
            timeout=600,
        )
        return {"status": "success", "result": result}

    except asyncio.TimeoutError:
        return {"status": "error", "message": "Processing timeout"}, 504
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {"status": "error", "message": str(e)}, 500