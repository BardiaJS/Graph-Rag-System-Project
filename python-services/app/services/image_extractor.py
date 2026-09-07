from pathlib import Path
import pymupdf


class ImageExtractor:

    def __init__(self, file_path, user_id, document_id):

        self.file_path = Path(file_path)
        self.user_id = user_id
        self.document_id = document_id

        self.images_path = (
            self.file_path.parent
            / str(self.document_id)
            / "images"
        )

        self.images_path.mkdir(parents=True, exist_ok=True)

    def image_extract(self):

        doc = pymupdf.open(self.file_path)

        images = []

        for page_index in range(len(doc)):

            page = doc[page_index]
            image_list = page.get_images()

            if image_list:
                print(
                    f"Found {len(image_list)} images "
                    f"on page {page_index + 1}"
                )
            else:
                print(
                    f"No images found on page {page_index + 1}"
                )

            for image_index, img in enumerate(image_list, start=1):

                xref = img[0]

                pix = pymupdf.Pixmap(doc, xref)

                if pix.n - pix.alpha > 3:
                    pix = pymupdf.Pixmap(
                        pymupdf.csRGB,
                        pix
                    )

                image_path = (
                    self.images_path
                    / f"page_{page_index + 1}-image_{image_index}.png"
                )

                pix.save(image_path)

                images.append({
                    "page_number": page_index + 1,
                    "image_number": image_index,
                    "path": str(image_path)
                })

                pix = None

        doc.close()

        return images
