import torch

# Load YOLOv5n model
print("Loading YOLOv5n...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.eval()

# Export to ONNX
print("Exporting to ONNX...")
dummy_input = torch.randn(1, 3, 320, 320)
torch.onnx.export(
    model, 
    dummy_input, 
    "yolov5n.onnx",
    opset_version=11,
    input_names=['images'],
    output_names=['output']
)

# Create class names file
print("Creating class_names.txt...")
with open("class_names.txt", "w") as f:
    for i in range(len(model.names)):
        f.write(f"{model.names[i]}\n")

print("Done! Created:")
print("- yolov5n.onnx")
print("- class_names.txt")