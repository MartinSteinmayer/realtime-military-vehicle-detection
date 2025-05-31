import os
import torch
from PIL import Image
from GroundingDINO.groundingdino.util.inference import load_model, predict
from torchvision.transforms import ToTensor

from utils import *

# Set paths

# Model setup
CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
CHECKPOINT_PATH = "groundingdino_swint_ogc.pth"  # Download from model zoo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def convert_bbox_to_yolo(box, img_w, img_h):
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0 / img_w
    y_center = (y_min + y_max) / 2.0 / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    return x_center, y_center, width, height


def run_groundingdino(config : dict) -> bool:
    # TODO
    return True