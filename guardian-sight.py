from ultralytics import YOLO

from utils import *
from dino_annotate import process_dataset


def train_yolo(config: dict):

    specs = get_available_specs()

    yolo_model = YOLO(config["model"])
    results = yolo_model.train(
        data=(config["data_yaml"]),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"] if specs["cuda_available"] else "cpu"
    )
    
    print("Training completed successfully.")


def main():
    config = load_config("config.json")
    if not config:
        print("Could not run config.")
        return
    
    run_dino = config["run_dino"]
    if run_dino:
        dino_cfg = config["groundingdino"]
        process_dataset(dino_cfg)
    
    #yolo_cfg = config["yolo_training"]
    #train_yolo(yolo_cfg)


# Run the program
if __name__ == "__main__":
    print("Starting the Guardian Sight program...")
    main()