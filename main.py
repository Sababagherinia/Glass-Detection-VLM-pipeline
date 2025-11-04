from object_detection import detect_objects, visualize_detections
# Example usage
image_path = "path/to/your/image.jpg"
detections = detect_objects(image_path, threshold=0.3)
visualize_detections(image_path, detections)