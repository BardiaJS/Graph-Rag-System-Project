import pymupdf
import os

class ImageExtractor:
    def __init__(self, file_path, output_dir="extracted_images"):
        self.file_path = file_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.doc_name = os.path.splitext(os.path.basename(file_path))[0]

    def extract_image(self):
        doc = pymupdf.open(self.file_path)
        number = 0
        for page in doc:
            images = page.get_images(full=True)
            print(f"Page {page.number}: {len(images)} images found")   # ← این رو اضافه کن
            for img_info in images:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                ext = base_image["ext"]
                filename = f"{self.doc_name}_page-{page.number}_img-{number}.{ext}"
                output_path = os.path.join(self.output_dir, filename)
                with open(output_path, "wb") as f:
                    f.write(base_image["image"])
                print(f"Saved: {output_path}")   # ← این رو هم اضافه کن
                number += 1
        doc.close()
        print(f"Total: {number}")
        return number