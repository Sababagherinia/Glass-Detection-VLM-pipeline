"""vlm.py

Simple CLIP-style VLM wrapper using Hugging Face `transformers` (CLIPModel + CLIPProcessor).

Provides image and text embedding helpers and cosine similarity utilities.
"""
from typing import List
import numpy as np
import torch

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class VLM:
    def __init__(self, device: str | None = None, model_name: str = "openai/clip-vit-base-patch32"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def image_embedding(self, image: Image.Image) -> np.ndarray:
        """Return a 1 x D numpy image embedding."""
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            img_feats = self.model.get_image_features(**inputs)
        img_feats = img_feats.cpu().numpy()
        # normalize
        img_feats = img_feats / np.linalg.norm(img_feats, axis=1, keepdims=True)
        return img_feats

    def text_embeddings(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            txt_feats = self.model.get_text_features(**inputs)
        txt_feats = txt_feats.cpu().numpy()
        txt_feats = txt_feats / np.linalg.norm(txt_feats, axis=1, keepdims=True)
        return txt_feats

    @staticmethod
    def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between rows of `a` and rows of `b`.

        Returns an (a.shape[0], b.shape[0]) array.
        """
        # a and b are expected to be normalized
        return np.matmul(a, b.T)
