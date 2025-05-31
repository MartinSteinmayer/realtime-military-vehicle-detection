import os
import torch
import torchvision
from PIL import Image
from GroundingDINO.groundingdino.util.inference import load_model, predict
from torchvision.transforms import ToTensor
import numpy as np

# Set paths
ROOT_DIR = "dataset"
SPLITS = ["train", "test", "validation"]
OUTPUT_DIR = "yolo_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model setup
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "groundingdino_swint_ogc.pth"  # Download from model zoo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
model.to(DEVICE)

# List of all classes (used as text prompts)
CLASSES = [
    "Anti-aircraft", "Armored personnel carriers", "Light armored vehicles",
    "Mine-protected vehicles", "Self-propelled artillery", "Armored combat support vehicles",
    "Infantry fighting vehicles", "light utility vehicles", "Prime movers and trucks", "tanks"
]

def convert_bbox_to_yolo(box, img_w, img_h):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height

# Main loop
for split in SPLITS:
    input_split_path = os.path.join(ROOT_DIR, split)
    output_images_path = os.path.join(OUTPUT_DIR, split, "images")
    output_labels_path = os.path.join(OUTPUT_DIR, split, "labels")
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    for class_id, class_name in enumerate(CLASSES):
        class_path = os.path.join(input_split_path, class_name)
        for img_file in os.listdir(class_path):
            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(class_path, img_file)
            output_img_path = os.path.join(output_images_path, img_file)
            label_path = os.path.join(output_labels_path, os.path.splitext(img_file)[0] + ".txt")

            # Load and transform image
            image_pil = Image.open(image_path).convert("RGB")
            image_tensor = ToTensor()(image_pil).unsqueeze(0).to(DEVICE)
            image_w, image_h = image_pil.size

            # Predict
            boxes, logits, phrases = predict(model, image_pil, class_name, box_threshold=0.3, text_threshold=0.25)

            # Save image
            image_pil.save(output_img_path)

            # Write label file
            with open(label_path, "w") as f:
                for box in boxes:
                    box = box.tolist()
                    x_center, y_center, width, height = convert_bbox_to_yolo(box, image_w, image_h)
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
