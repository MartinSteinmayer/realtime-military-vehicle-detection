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
        text_threshold = text_threshold,
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
    )
    return boxes, logits, phrases


def process_single_image(dino, image_path, output_dirs, class_id, text_prompt, box_threshold=0.4, text_threshold=0.1, use_cache=True):
    images_dir, labels_dir, annotated_dir = output_dirs

    # Derive base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_filename = f"{base_name}.txt"
    label_path = os.path.join(labels_dir, label_filename)

    # Skip if label file exists and is non-empty
    if use_cache and os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        return (image_path, True)

    try:
        image_source, image = load_image(image_path)
        boxes, logits, phrases = run_dino(dino, image, text_prompt, box_threshold, text_threshold)

        if len(boxes) == 0:
            print(f"No detections in {image_path}, skipping...")
            return (image_path, False)

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        h, w = image_source.shape[:2]
        
        image_filename = f"{base_name}.jpg"

        # Save outputs
        cv2.imwrite(os.path.join(images_dir, image_filename), image_source)
        cv2.imwrite(os.path.join(annotated_dir, image_filename), annotated_frame)

        yolo_boxes = convert_to_yolo_format(boxes, w, h)
        with open(label_path, 'w') as f:
            for box in yolo_boxes:
                f.write(f"{class_id} {' '.join([str(coord) for coord in box])}\n")

        return (image_path, True)

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return (image_path, False)


def process_folder_parallel(dino, num_workers, folder_path, class_id, box_threshold=0.4, text_threshold=0.1, output_dir='output'):
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
                             text_threshold=text_threshold,
                             use_cache=True)    # We want to skip pre-annotated images so we set use_cache as True and check if the label exists
        
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


def process_dataset(config: dict):
    """Process the entire dataset sequentially (one folder at a time)"""

    dataset_path = config["dataset"]
    output_path = config["output_dir"]

    specs = get_available_specs()
    num_workers = min(specs["cpu_cores"], config["num_workers"])

    # Load the model once
    model = load_model(config["model_config_path"], config["model_checkpoint_path"], device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check directory structure
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist or is not a directory") 
    
    class_names = fetch_classes(dataset_path)
    if not class_names:
        raise ValueError(f"Could not fetch class names for the given dataset: {dataset_path}")
    
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
            process_folder_parallel(dino=model, num_workers=num_workers,
                                    folder_path=folder_path,
                                    class_id=class_id,
                                    box_threshold=0.4, 
                                    text_threshold=0.25, 
                                    output_dir=output_dir, 
                                    )
            
            if folder not in class_names:
                class_names.append(folder)
    
    # Create YAML config for YOLO training
    yaml_config = {
        'names': class_names,
        'nc': len(class_names),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images'
    }
    
    output_yaml_path = os.path.join(output_path, "data.yaml")
    with open(output_yaml_path, 'w') as f:
        yaml.dump(yaml_config, f)
    
    print(f"\nDataset processed. Config saved to {output_yaml_path}")
    print(f"Found classes: {class_names}")