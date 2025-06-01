#!/usr/bin/env python3
"""
Script to download, verify, and export YOLO8n model in ONNX format
Optimized for OpenCV 4.6 compatibility
"""

from ultralytics import YOLO
import cv2
import numpy as np
import onnx
import os

def main():
    print("=== YOLO8n Model Export Script (OpenCV 4.6 Compatible) ===\n")
    
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
    
    # 4. Export to ONNX format with OpenCV 4.6 compatibility settings
    print("\n4. Exporting to ONNX format (OpenCV 4.6 compatible)...")
    
    # Delete old model if it exists
    old_model_path = "yolov8n.onnx"
    if os.path.exists(old_model_path):
        os.remove(old_model_path)
        print(f"   ✓ Removed old model: {old_model_path}")
    
    try:
        export_path = model.export(
            format="onnx",
            imgsz=320,          # FIXED: Use 320 to match your C++ code
            dynamic=False,      # Fixed input size for better compatibility
            simplify=True,      # Simplify the model for better performance
            opset=11,          # ONNX opset version for OpenCV 4.6 compatibility
            half=False,        # Use FP32 instead of FP16 for better compatibility
            int8=False,        # Disable quantization
            optimize=False,    # Disable optimization that might cause issues
            verbose=True       # Show export details
        )
        print(f"   ✓ Model exported to: {export_path}")
        
        # Verify the ONNX model
        print("\n   Verifying exported ONNX model...")
        onnx_model = onnx.load(export_path)
        
        # Check opset version
        opset_version = onnx_model.opset_import[0].version
        print(f"   ✓ ONNX opset version: {opset_version}")
        
        if opset_version <= 13:
            print("   ✓ Opset version is compatible with OpenCV 4.6")
        else:
            print(f"   ⚠ Warning: Opset {opset_version} might be too new for OpenCV 4.6")
        
        # Check model shapes
        input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]
        
        print(f"   ✓ Input shape: {input_shape}")
        print(f"   ✓ Output shape: {output_shape}")
        
        # Verify expected shapes
        if input_shape == [1, 3, 320, 320]:
            print("   ✓ Input shape matches expected [1, 3, 320, 320]")
        else:
            print(f"   ⚠ Warning: Unexpected input shape {input_shape}")
            
        if output_shape[0] == 1 and output_shape[1] == 84:
            print(f"   ✓ Output shape looks correct: [1, 84, {output_shape[2]}]")
        else:
            print(f"   ⚠ Warning: Unexpected output shape {output_shape}")
            
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return
    
    # 5. Create class names file
    print("\n5. Creating class_names.txt file...")
    with open("class_names.txt", "w") as f:
        for i in range(len(model_names)):
            f.write(f"{model_names[i]}\n")
    print("   ✓ class_names.txt created with correct COCO class names")
    
    # 6. Test the exported model with OpenCV DNN using correct dimensions
    print("\n6. Testing exported model with OpenCV DNN...")
    try:
        # Check OpenCV version
        opencv_version = cv2.__version__
        print(f"   OpenCV version: {opencv_version}")
        
        # Load the ONNX model with OpenCV
        net = cv2.dnn.readNetFromONNX(export_path)
        print("   ✓ Model loaded with OpenCV DNN")
        
        # Set backend to CPU for maximum compatibility
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("   ✓ Backend set to CPU")
        
        # FIXED: Create a dummy input with correct size (320x320, not 640x640)
        dummy_input = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(
            dummy_input, 
            1.0/255.0,          # Scale factor
            (320, 320),         # FIXED: Use 320x320 to match export
            (0, 0, 0),          # Mean subtraction (none)
            swapRB=True,        # BGR to RGB
            crop=False,         # Don't crop
            ddepth=cv2.CV_32F   # Float32
        )
        
        print(f"   ✓ Created test blob with shape: {blob.shape}")
        
        # Run inference
        net.setInput(blob)
        print("   ✓ setInput successful")
        
        # Try different forward methods
        outputs = None
        try:
            # Method 1: Simple forward
            outputs = [net.forward()]
            print("   ✓ Simple forward() successful")
        except Exception as e1:
            print(f"   Simple forward failed: {e1}")
            try:
                # Method 2: Forward with output names
                output_names = net.getUnconnectedOutLayersNames()
                outputs = []
                net.forward(outputs, output_names)
                print("   ✓ Named forward() successful")
            except Exception as e2:
                print(f"   Named forward failed: {e2}")
                raise e2
        
        if outputs and len(outputs) > 0:
            output = outputs[0]
            print(f"   ✓ OpenCV DNN test successful")
            print(f"   ✓ Output shape: {output.shape}")
            print(f"   ✓ Output dimensions: {output.ndim}D")
            print(f"   ✓ Output dtype: {output.dtype}")
            
            # Verify output format
            if output.ndim == 3 and output.shape[0] == 1 and output.shape[1] == 84:
                print(f"   ✓ Output format is correct: [1, 84, {output.shape[2]}]")
                print("   ✓ Model is compatible with your C++ code!")
            else:
                print(f"   ⚠ Warning: Unexpected output format {output.shape}")
        else:
            print("   ✗ No outputs returned")
            
    except Exception as e:
        print(f"   ✗ OpenCV test failed: {e}")
        print("   This indicates the model is not compatible with your OpenCV version")
    
    # 7. Create a proper test image with correct size
    print("\n7. Creating test verification...")
    # FIXED: Create 320x320 test image to match model input
    test_img = np.ones((320, 320, 3), dtype=np.uint8) * 128  # Gray image
    cv2.putText(test_img, "YOLO8n", (80, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(test_img, "320x320", (100, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite("test_image_320.jpg", test_img)
    print("   ✓ test_image_320.jpg created for verification")
    
    print("\n=== Export Complete ===")
    print("Files created:")
    print(f"  • {export_path} (ONNX model - OpenCV 4.6 compatible)")
    print("  • class_names.txt (class names)")
    print("  • test_image_320.jpg (320x320 test image)")
    print("\nYou can now use these files in your C++ application!")
    
    # 8. Print summary for C++ integration
    print("\n8. C++ Integration Summary:")
    print("   Constants to verify in your C++ code:")
    print("   #define INPUT_WIDTH 320")
    print("   #define INPUT_HEIGHT 320")
    print("   #define CONFIDENCE_THRESHOLD 0.5f")
    print("   #define NMS_THRESHOLD 0.4f")
    print("")
    print("   Expected model output: [1, 84, 2100] (approximately)")
    print("   - 84 = 4 (bbox) + 80 (COCO classes)")
    print("   - 2100 = number of detections (varies by model)")

if __name__ == "__main__":
    main()