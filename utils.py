import os
import json

def fetch_classes(dataset_path: str) -> list[str] | None:
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


# Loads a config from a json file
def load_config(config_path: str = "config.json") -> dict:
    if not os.path.exists(config_path):
        print(f"Invalid path specified for config: {config_path}")
        return None
    
    try:
        with open(config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)
            for item in ["dataset", "output_dir"]:
                if item not in config:
                    print(f"Missing config parameter: {item}")
                    return None
            return config

    except Exception as e:
        print(f"Could not load config: {e}")
        return None