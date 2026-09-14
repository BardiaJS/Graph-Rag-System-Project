import pymupdf

class TextExtractor:
    def __init__(self, file_path):
        self.file_path = file_path


    def has_usable_text(self, text: str) -> bool:
        """
        بررسی می‌کند که آیا متن استخراج‌شده از صفحه
        برای استفاده مناسب است یا باید OCR انجام شود.

        Returns:
            True  -> متن قابل استفاده است
            False -> متن قابل استفاده نیست و باید OCR شود
        """

        # text not exist
        if not text:
            return False

        # delete all spaces
        text = text.strip()

        # if not anything exist
        if not text:
            return False

        # text is very short
        if len(text) < 50:
            return False

        # number of digits and letters
        alnum_count = sum(
            character.isalnum()
            for character in text
        )

        # at least 30% of text should be letters
        alnum_ratio = alnum_count / len(text)

        if alnum_ratio < 0.3:
            return False

        # all success
        return True
    def count_page_number(self ):
        doc = pymupdf.open(self.file_path)
        print(doc.page_count)

    def get_metadata (self ):
        doc = pymupdf.open(self.file_path)
        print(doc.metadata)

    def load_pages(self):
        doc = pymupdf.open(self.file_path)
        pages = []

        for page_number, page in enumerate(doc, start=1):

            # استخراج متن معمولی
            text = page.get_text()

            # بررسی قابل استفاده بودن متن
            if self.has_usable_text(text):

                print(f"\nPage {page_number}: NORMAL TEXT")

                source = "text"

            else:

                print(f"\nPage {page_number}: OCR")

                # OCR با Tesseract
                textpage = page.get_textpage_ocr(
                    language="eng",
                    dpi=300,
                    full=True
                )

                # استخراج متن OCR شده
                text = page.get_text(
                    textpage=textpage
                )

                source = "ocr"

            print(
                f"-------------------------------------------\n"
                f"{text}\n"
            )

            pages.append({
                "page_number": page_number,
                "text": text,
                "source": source
            })

        doc.close()

        return pages
        

    def extract_blocks(self):
        doc = pymupdf.open(self.file_path)
        for page in doc:
            print('---------------------------------------------- \n')
            print("blocks are: \n")
            text = page.get_text("blocks")
            print(text)

    def extract_words(self):
        doc = pymupdf.open(self.file_path)
        for page in doc:
            print('********************************************** \n')
            print("words are: \n")
            text = page.get_text("words")
            print(text)


    # def extract_dict(self):
    #     doc = pymupdf.open(self.file_path)
    #     for page in doc:
    #         print('+++++++++++++++++++++++++++++++++++++++++++++++ \n')
    #         print("dict are: \n")
    #         text = page.dic
    #         print(text)