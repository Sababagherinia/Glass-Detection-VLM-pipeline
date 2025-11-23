from detector import OwlDetector
from semantic_safety_map import build_safety_map, plot_safety_map
from PIL import Image, ImageDraw
import glob

detector = OwlDetector()

# loading a bunch of images with glass objects

for path in glob.glob("D:/uni_vub/thesis/Glass-Detection-VLM-pipeline/data/rgb/*"):
    print("\n==========================================")
    print(f"Processing: {path}")
    img = Image.open(path).convert("RGB")

    queries = ["a glass window",
        "a transparent surface",
        "a glass wall",
        "a mirror",
        "window",]

    detections = detector.detect(img, queries)

# create bb in the detected image
    print(len(detections))
    for det in detections:
        print(f"Detected {det['label']} with score {det['score']:.2f} at box {det['box']}")
        img = img.copy()
        draw = ImageDraw.Draw(img)
        x0, y0, x1, y1 = det["box"]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    img.show()

    if len(detections) > 0:
        grid, decision = build_safety_map(
            detections,
            image_width=img.size[0],
            image_height=img.size[1]
        )

        print("Safety decision:", decision)

        # plot_safety_map(grid, decision)
    else:
        print("No detections to build safety map.")
