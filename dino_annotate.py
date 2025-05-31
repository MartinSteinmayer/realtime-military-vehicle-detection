from datasets import load_dataset
import cv2
import os
import yaml
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from utils import *

# Limit OpenMP threads to prevent thread explosion
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate


def run_dino(dino, image, text_prompt='placeholder', box_threshold=0.4, text_threshold=0.1):
    boxes, logits, phrases = predict(
        model = dino, 
        image = image, 
        caption = text_prompt, 
        box_threshold = box_threshold, 
        text_threshold = text_threshold
    )
    return boxes, logits, phrases


def convert_to_yolo_format(boxes, image_width, image_height):
    """Convert boxes from [x1, y1, x2, y2] to YOLO [x_center, y_center, width, height] format"""
    yolo_boxes = []
    for box in boxes.tolist():
        x1, y1, x2, y2 = box
    
        # Normalize to 0-1 range
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        
        yolo_boxes.append([x_center, y_center, width, height])
        
    return yolo_boxes


def process_single_image(dino, image_path, output_dirs, class_id, text_prompt, box_threshold=0.4, text_threshold=0.1):
    """Process a single image with DINO"""
    images_dir, labels_dir, annotated_dir = output_dirs
    
    try:
        # Load and process image
        image_source, image = load_image(image_path)
        boxes, logits, phrases = run_dino(dino, image, text_prompt, box_threshold, text_threshold)
        
        if len(boxes) == 0:
            print(f"No detections in {image_path}, skipping...")
            return (image_path, False)
            
        # Create annotated image
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        # Get image dimensions for YOLO conversion
        h, w = image_source.shape[:2]
        
        # Create filenames - use original filename without extension as base
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_filename = f"{base_name}.jpg"
        label_filename = f"{base_name}.txt"
        
        # Save image, annotated image, and label
        cv2.imwrite(os.path.join(images_dir, image_filename), image_source)
        cv2.imwrite(os.path.join(annotated_dir, image_filename), annotated_frame)
        
        # Convert boxes to YOLO format and save labels
        yolo_boxes = convert_to_yolo_format(boxes, w, h)
        with open(os.path.join(labels_dir, label_filename), 'w') as f:
            for box in yolo_boxes:
                # Format: class_id x_center y_center width height
                f.write(f"{class_id} {' '.join([str(coord) for coord in box])}\n")
        
        return (image_path, True)
                
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return (image_path, False)


def process_folder_sequential(dino, folder_path, class_id, box_threshold=0.4, text_threshold=0.1, output_dir='output', num_workers=20):
    """Process a folder of images with limited parallelism"""
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create output directories
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    annotated_dir = os.path.join(output_dir, 'annotated')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    
    # Get class name from folder
    class_name = os.path.basename(folder_path)
    text_prompt = f"{class_name}"
    
    output_dirs = (images_dir, labels_dir, annotated_dir)
    
    print(f"Processing {len(images)} images for {class_name}")
    
    # Process with limited parallelism
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Create partial function with fixed parameters
        process_func = partial(process_single_image, dino, 
                             output_dirs=output_dirs,
                             class_id=class_id,
                             text_prompt=text_prompt, 
                             box_threshold=box_threshold, 
                             text_threshold=text_threshold)
        
        # Process images with progress bar
        with tqdm(total=len(images), desc=f"Processing {class_name}") as pbar:
            futures = []
            for image_path in images:
                future = executor.submit(process_func, image_path)
                futures.append(future)
            
            # Collect results as they complete
            for future in futures:
                result = future.result()
                pbar.update(1)
    
    print(f"Finished processing {class_name}")


def process_dataset():
    """Process the entire dataset sequentially (one folder at a time)"""

    # Load dataset path, output path, and the number of workers from the config
    config = load_config("config.json")

    dataset_path = config["dataset"]
    output_path = config["output_dir"]  
    num_workers = config["num_workers"]

    # Load the model once
    model = load_model(config["model_config_path"], config["model_checkpoint_path"])
    
    # Check directory structure
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist or is not a directory") 
    
    class_names = fetch_classes(dataset_path)
    
    # Process train, val, test folders if they exist
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_path, split)
        if not os.path.isdir(split_dir):
            print(f"Split directory {split_dir} not found, skipping...")
            continue
            
        # Process each class folder sequentially
        class_folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        for class_id, folder in enumerate(class_folders):
            folder_path = os.path.join(split_dir, folder)
            print(f"\nProcessing split: {split}, folder: {folder}")
            
            output_dir = os.path.join(output_path, split, folder)
            # Process folder
            process_folder_sequential(model, folder_path,
                                    class_id=class_id,
                                    box_threshold=0.4, 
                                    text_threshold=0.25, 
                                    output_dir=output_dir, 
                                    num_workers=num_workers)
            
            if folder not in class_names:
                class_names.append(folder)
    
    # Create YAML config for YOLO training
    config = {
        'names': class_names,
        'nc': len(class_names),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images'
    }
    
    output_yaml_path = os.path.join(output_path, "data.yaml")
    with open(output_yaml_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\nDataset processed. Config saved to {output_yaml_path}")
    print(f"Found classes: {class_names}")


if __name__ == "__main__":
    # Process with limited parallelism to avoid resource exhaustion
    # Adjust num_workers in the config file based on your system's capabilities
    process_dataset()