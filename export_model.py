#!/usr/bin/env python3
"""
Script to download, verify, and export YOLO8n model in ONNX format
This ensures we have the correct model with proper class encoding
"""

from ultralytics import YOLO
import cv2
import numpy as np

def main():
    print("=== YOLO8n Model Export Script ===\n")
    
    # 1. Load the official YOLO8n model (will download if not exists)
    print("1. Loading YOLO8n model...")
    model = YOLO("yolov8n.pt")
    print("   ✓ Model loaded successfully")
    
    # 2. Print model information
    print("\n2. Model Information:")
    print(f"   Model type: {model.task}")
    
    # 3. Verify the class names match COCO dataset
    print("\n3. Verifying class names (first 10):")
    expected_classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", 
        "bus", "train", "truck", "boat", "traffic light"
    ]
    
    model_names = model.names
    print(f"   Total classes: {len(model_names)}")
    
    for i in range(min(10, len(model_names))):
        class_name = model_names[i]
        expected = expected_classes[i] if i < len(expected_classes) else "N/A"
        match = "✓" if class_name == expected else "✗"
        print(f"   {i:2d}: {class_name:15s} (expected: {expected:15s}) {match}")
    
    # 4. Export to ONNX format
    print("\n4. Exporting to ONNX format...")
    export_path = model.export(
        format="onnx",
        imgsz=640,  # Input image size
        dynamic=False,  # Fixed input size for better compatibility
        simplify=True,  # Simplify the model for better performance
        opset=11  # ONNX opset version (good compatibility)
    )
    print(f"   ✓ Model exported to: {export_path}")
    
    # 5. Create class names file
    print("\n5. Creating class_names.txt file...")
    with open("class_names.txt", "w") as f:
        for i in range(len(model_names)):
            f.write(f"{model_names[i]}\n")
    print("   ✓ class_names.txt created with correct COCO class names")
    
    # 6. Test the exported model with OpenCV DNN
    print("\n6. Testing exported model with OpenCV DNN...")
    try:
        # Load the ONNX model with OpenCV
        net = cv2.dnn.readNetFromONNX(export_path)
                
        # Create a dummy input to test model dimensions
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(dummy_input, 1.0/255.0, (640, 640), swapRB=True, crop=False)
        
        # Run inference
        net.setInput(blob)
        outputs = net.forward()
        
        # Print output information
        print(f"   ✓ OpenCV DNN test successful")
        print(f"   ✓ Output shape: {outputs[0].shape}")
        print(f"   ✓ Output dimensions: {outputs[0].ndim}D")
        
        # Expected shape for YOLO8n: (1, 84, 8400) 
        # - 1: batch size
        # - 84: 4 (bbox) + 80 (classes) 
        # - 8400: number of detections
        expected_attrs = 4 + len(model_names)  # 4 bbox + 80 classes = 84
        actual_attrs = outputs[0].shape[1] if outputs[0].ndim >= 2 else 0
        
        print(f"   ✓ Expected attributes: {expected_attrs}")
        print(f"   ✓ Actual attributes: {actual_attrs}")
        
        if actual_attrs == expected_attrs:
            print("   ✓ Model output format is correct!")
        else:
            print("   ⚠ Warning: Unexpected output format")
            
    except Exception as e:
        print(f"   ✗ OpenCV test failed: {e}")
    
    # 7. Create a verification image test (optional)
    print("\n7. Creating test verification...")
    test_img = np.ones((640, 640, 3), dtype=np.uint8) * 128  # Gray image
    cv2.putText(test_img, "YOLO8n Test Image", (50, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imwrite("test_image.jpg", test_img)
    print("   ✓ test_image.jpg created for verification")
    
    print("\n=== Export Complete ===")
    print("Files created:")
    print(f"  • {export_path} (ONNX model)")
    print("  • class_names.txt (class names)")
    print("  • test_image.jpg (test image)")
    print("\nYou can now use these files in your C++ application!")
    
    # 8. Print the exact class mapping for verification
    print("\n8. Complete class mapping:")
    print("   Index | Class Name")
    print("   ------|------------")
    for i, name in model_names.items():
        print(f"   {i:5d} | {name}")

if __name__ == "__main__":
    main()