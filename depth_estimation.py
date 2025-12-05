import os
import torch
from transformers import pipeline
from PIL import Image

# load pipe, use gpu if available
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if torch.cuda.is_available() else -1, use_fast=True)

# load all images from rgb folder


for filename in os.listdir("data/rgb"):
    image = Image.open(f"data/rgb/{filename}")

    # inference
    depth = pipe(image)["depth"]

    # saving depth map colored for visualization
    depth.save(f"data/depth/{filename}_depth.png")
    print(f"Depth map for {filename} saved to depth folder")