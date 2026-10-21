from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer


class ChunkingService:

    def __init__(self , file_path):
        self.file_path = file_path

    def chunk(self):
        # ۱. تبدیل PDF
        converter = DocumentConverter()
        result = converter.convert(self.file_path)
        doc = result.document

        # ۲. Tokenizer هماهنگ با مدل embedding
        EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
            max_tokens=256,  # ← سقف واقعی این مدل
        )

        # ۳. Chunker
        chunker = HybridChunker(
            tokenizer=tokenizer,
            merge_peers=True,          # ادغام chunkهای کوچیک
            repeat_table_header=True,  # تکرار هدر جدول در chunkهای بعدی
        )

        # ۴. Chunking
        chunks = list(chunker.chunk(dl_doc=doc))

        # ۵. استفاده برای embedding
        for chunk in chunks:
            # متن خام (برای نمایش/ذخیره)
            raw_text = chunk.text
            
            # متن غنی‌شده با heading (برای embedding) ← مهم!
            enriched_text = chunker.contextualize(chunk=chunk)
            
            # metadata
            headings = chunk.meta.headings if chunk.meta else []
            page = chunk.meta.page_no if chunk.meta else None
            
            # embedding روی متن غنی‌شده
            embedding = embed_model.encode(enriched_text)