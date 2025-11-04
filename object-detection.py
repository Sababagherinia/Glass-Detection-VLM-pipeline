#implementing object detection using a CLIP pre-trained model from huggingface transformers
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the pre-trained CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Function to perform object detection
def detect_objects(image_path, threshold=0.3):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # Define a set of object categories
    categories = ["cat", "dog", "car", "person", "bicycle", "tree", "building", "flower", "chair", "table"]

    # Generate candidate bounding boxes (for simplicity, using a grid approach)
    boxes = []
    box_size = 100
    step_size = 50
    for y in range(0, height - box_size, step_size):
        for x in range(0, width - box_size, step_size):
            boxes.append((x, y, x + box_size, y + box_size))

    detected_objects = []

    for box in boxes:
        cropped_image = image.crop(box)
        inputs = processor(text=categories, images=cropped_image, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        max_prob_index = np.argmax(probs)
        max_prob = probs[max_prob_index]

        if max_prob > threshold:
            detected_objects.append({
                "category": categories[max_prob_index],
                "box": box,
                "confidence": max_prob
            })

    return detected_objects
# Function to visualize detected objects
def visualize_detections(image_path, detections):
    image = cv2.imread(image_path)
    for detection in detections:
        box = detection["box"]
        category = detection["category"]
        confidence = detection["confidence"]

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        label = f"{category}: {confidence:.2f}"
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show() 