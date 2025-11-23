"""detector.py

OWL-ViT based open-vocabulary detector wrapper using Hugging Face `transformers`.

Provides a simple `OwlDetector` class that returns boxes, scores and query labels for an input PIL image.
"""
from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection


class OwlDetector:
    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)

    def detect(self, image: Image.Image, queries: List[str], threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Run open-vocabulary detection for the provided queries on the image.

        Returns a list of detection dictionaries: {"box": (x0,y0,x1,y1), "score": float, "label": str}
        """
        # Prepare inputs
        inputs = self.processor(images=image, text=queries, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process to get boxes and scores in image coordinates
        target_sizes = torch.tensor([(image.size[1], image.size[0])], dtype=torch.long)
        results = self.processor.post_process_grounded_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)

        detections: List[Dict[str, Any]] = []
        # results is a list (batch)
        for res in results:
            boxes = res.get("boxes")
            scores = res.get("scores")
            labels = res.get("labels")
            if boxes is None:
                continue
            for box, score, lab in zip(boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()):
                # box is [x0, y0, x1, y1]
                detections.append({"box": tuple(map(float, box.tolist())), "score": float(score), "label": queries[int(lab)]})

        # sort by score desc
        detections.sort(key=lambda x: x["score"], reverse=True)
        return detections
