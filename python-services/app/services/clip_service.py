from typing import List

import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
)


class CLIPService:
    """
    تولید embedding برای تصاویر با CLIP.
    """

    MODEL_NAME = (
        "openai/clip-vit-base-patch32"
    )

    VECTOR_SIZE = 512

    def __init__(
        self,
        device: str = "cpu",
    ):

        self.device = device

        print(
            "🖼️ Loading CLIP model..."
        )

        self.model = CLIPModel.from_pretrained(
            self.MODEL_NAME
        )

        self.processor = CLIPProcessor.from_pretrained(
            self.MODEL_NAME
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"   ✅ CLIP loaded on "
            f"{self.device}"
        )

    def get_image_embedding(
        self,
        image_path: str,
    ) -> List[float]:

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(
                    self.device
                )
                for key, value in inputs.items()
            }

            with torch.no_grad():

                features = (
                    self.model.get_image_features(
                        **inputs
                    )
                )

                features = torch.nn.functional.normalize(
                    features,
                    p=2,
                    dim=-1,
                )

            return (
                features
                .cpu()
                .numpy()
                .flatten()
                .tolist()
            )

        except Exception as e:

            print(
                f"⚠️ CLIP image embedding failed: {e}"
            )

            return []