import fitz  # PyMuPDF
from typing import List, Dict, Any
import json
from pathlib import Path

from app.services.document_processor import PYMUPDF_AVAILABLE

class PDFTextExtractor:
    """استخراج هوشمند متن از PDF با حفظ ساختار دوستونه"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    
    async def _extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """استخراج متن و تصاویر از PDF"""
        if not PYMUPDF_AVAILABLE:
            return {
                'pages': 0,
                'full_text': '',
                'figures': [],
                'tables': []
            }
        
        try:
            print(f"   📖 Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            print(f"   📄 Number of pages: {len(doc)}")
            
            full_text = ""
            figures = []
            tables = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                full_text += text + "\n\n"
                
                # استخراج تصاویر
                images = page.get_images(full=True)
                for img in images:
                    figures.append({
                        'page': page_num + 1,
                        'index': len(figures) + 1
                    })
            
            # بستن بعد از استخراج کامل
            doc.close()
            
            print(f"   📝 Extracted {len(figures)} figures")
            print(f"   📝 Full text length: {len(full_text)} characters")
            
            return {
                'pages': len(doc),
                'full_text': full_text,
                'figures': figures,
                'tables': tables
            }
            
        except Exception as e:
            print(f"   ⚠️ Error extracting text: {e}")
            import traceback
            traceback.print_exc()
            return {
                'pages': 0,
                'full_text': '',
                'figures': [],
                'tables': []
            }
    
    def _detect_columns(self, page) -> List[Dict]:
        """تشخیص ستون‌های صفحه با تحلیل موقعیت بلوک‌های متنی"""
        blocks = page.get_text("dict")['blocks']
        
        if not blocks:
            return [{'x0': 0, 'x1': page.rect.width, 'width': page.rect.width}]
        
        # پیدا کردن مرزهای عمودی بلوک‌ها
        x_positions = []
        for block in blocks:
            if 'lines' in block and block['lines']:
                x0 = block['bbox'][0]
                x1 = block['bbox'][2]
                x_positions.append((x0, x1))
        
        if not x_positions:
            return [{'x0': 0, 'x1': page.rect.width, 'width': page.rect.width}]
        
        # مرتب‌سازی و تشخیص خوشه‌های ستون‌ها
        centers = [(x0 + x1) / 2 for x0, x1 in x_positions]
        centers.sort()
        
        # اگر اختلاف مرکزها زیاد بود، یعنی دو ستون داریم
        if len(centers) > 1 and (centers[-1] - centers[0]) > page.rect.width * 0.4:
            mid = (centers[0] + centers[-1]) / 2
            return [
                {'x0': 0, 'x1': mid, 'width': mid},
                {'x0': mid, 'x1': page.rect.width, 'width': page.rect.width - mid}
            ]
        
        return [{'x0': 0, 'x1': page.rect.width, 'width': page.rect.width}]
    
    def _extract_text_by_columns(self, page, columns: List[Dict]) -> List[Dict]:
        """استخراج متن بر اساس ستون‌های تشخیص داده شده"""
        page_text = []
        
        for col in columns:
            # تنظیم ناحیه استخراج برای هر ستون
            clip = fitz.Rect(col['x0'], 0, col['x1'], page.rect.height)
            text = page.get_text("text", clip=clip)
            
            # پاکسازی متن
            text = self._clean_text(text)
            
            if text.strip():
                page_text.append({
                    'column': columns.index(col) + 1,
                    'text': text
                })
        
        return page_text
    
    def _clean_text(self, text: str) -> str:
        """پاکسازی متن از نویزهای اضافی"""
        # حذف هدر و فوتر
        lines = text.split('\n')
        if len(lines) > 10:
            # حذف اولین و آخرین خطوط که ممکن است هدر/فوتر باشند
            lines = lines[1:-1]
        
        # حذف خطوط خالی اضافی
        text = '\n'.join([line for line in lines if line.strip()])
        
        # حذف کاراکترهای خاص
        text = text.replace('\x0c', '')  # صفحه جدید
        
        return text
    
    def _extract_images(self, page, page_num: int) -> List[Dict]:
        """استخراج تصاویر از صفحه"""
        images = []
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(self.doc, xref)
                
                if pix.n - pix.alpha < 4:  # تصویر رنگی
                    img_data = pix.tobytes("png")
                else:
                    img_data = pix.tobytes("png")
                
                images.append({
                    'page': page_num + 1,
                    'index': img_index + 1,
                    'data': img_data,
                    'width': pix.width,
                    'height': pix.height
                })
                
                pix = None
            except Exception as e:
                print(f"Error extracting image {img_index}: {e}")
                continue
        
        return images
    
    def _extract_tables(self, page, page_num: int) -> List[Dict]:
        """استخراج جداول (ساده)"""
        tables = []
        # این بخش نیاز به کتابخانه‌های تخصصی‌تر مثل Camelot دارد
        # فعلاً یک نمونه ساده برمی‌گردانیم
        
        # تشخیص جدول با الگوی خطوط
        text = page.get_text("text")
        if '|' in text or '\t' in text:
            tables.append({
                'page': page_num + 1,
                'type': 'simple',
                'content': text
            })
        
        return tables