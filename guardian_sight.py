from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Export the model
model.export(format='onnx', 
            batch=1, 
            device='cpu', 
            simplify=True, 
            imgsz=320, 
            dynamic=False,
            nms=True,
            optimize=True)
