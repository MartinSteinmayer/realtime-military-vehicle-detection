import os
import json
import multiprocessing
import torch

def fetch_classes(dataset_path: str) -> list[str] | None:
    """Fetches the classes in a dataset folder with the expected structure"""
    subdirectories = os.listdir(dataset_path)
    for subdir in subdirectories:
        if subdir.lower() not in ["train", "test", "split"]:
            print(f"Invalid subdirectory in dataset: '{subdir}'. The dataset must contain the following directories: train, test, split")
            return None
        
    train_classes = os.listdir(os.path.join(dataset_path, "train"))
    test_classes = os.listdir(os.path.join(dataset_path, "test"))
    validation_classes = os.listdir(os.path.join(dataset_path, "validation"))

    # Check that all of the subdirectories contain the same classes
    if train_classes == test_classes == validation_classes:
        return train_classes
    
    # If not return None
    return None


def load_config(config_path: str = "config.json") -> dict:
    """Loads a config from a json file"""
    if not os.path.exists(config_path):
        print(f"Invalid path specified for config: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)

            # Check general config
            for item in ["run_dino", "groundingdino", "yolo_training"]:
                if item not in config:
                    print(f"Missing config parameter: {item}")
                    return None
            
            # If run_dino is set, check that the config contains all of the necessary parameters
            if config["run_dino"]:
                dino_cfg = config["groundingdino"]
                for item in ["dataset", "output_dir", "model_config_path", "model_checkpoint_path", "num_workers"]:
                    if item not in dino_cfg:
                        print(f"Missing config parameter: {item}")
                        return None
            
            # Check parameters necessary for yolo_training
            yolo_cfg = config["yolo_training"]
            for item in ["model", "epochs", "imgsz", "batch", "device", "data_yaml"]:
                if item not in yolo_cfg:
                        print(f"Missing config parameter: {item}")
                        return None
                
            return config

    except Exception as e:
        print(f"Could not load config: {e}")
        return None


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


def get_available_specs() -> dict:
    """Returns a dictionary containing the number of available cores in the system and if CUDA is available or not"""
    return {
        "cpu_cores" : multiprocessing.cpu_count(),
        "cuda_available" : torch.cuda.is_available()
    }