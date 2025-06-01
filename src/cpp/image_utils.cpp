#include "image_utils.h"


bool downscaleImage(const cv::Mat& input, cv::Mat& output, const cv::Size& targetSize) {
    if (input.empty()) {
        std::cerr << "Error: Input image size is 0." << std::endl;
        return false;
    }

    if (targetSize.height <= 0 || targetSize.width <= 0) {
        std::cerr << "Error: target image size must have width & height > 0!" << std::endl;
        return false;
    }

    try {
        // cv::INTER_AREA is the recommended interpolation method for downscaling images
        cv::resize(input, output, targetSize, 0, 0, cv::INTER_AREA);
        return true;
    } catch (const cv::Exception &e) {
        std::cerr << "Error resizing image: " << e.what() << std::endl;
        return false;
    }
}


bool readImage(const std::string& imagePath, cv::Mat& outputImage) {
    outputImage = cv::imread(imagePath);
    if (outputImage.empty()) {
        std::cerr << "Failed to read image: " << imagePath << std::endl;
        return false;
    }
    if (outputImage.channels() != 3) {
        std::cerr << "Image must have three channels (RGB/BGR)." << std::endl;
        return false;
    }
    return true;
}


cv::Mat preProcessImage(const std::string& imagePath, cv::Size& originalSize, cv::Size& readSize, cv::Point& offset) {
    cv::Mat inputImage;
    cv::Mat outputImage;

    if (!readImage(imagePath, inputImage)) {
        std::cerr << "No image read." << std::endl;
        return outputImage;
    }

    // Store the original image size BEFORE any processing
    originalSize = inputImage.size();

    if (!downscaleImage(inputImage, outputImage, cv::Size(INPUT_WIDTH, INPUT_HEIGHT))) {
        std::cerr << "Image couldn't be downscaled." << std::endl;
        return outputImage;
    }

    readSize = outputImage.size();

    int col = outputImage.cols;
    int row = outputImage.rows;
    int maxDimension = std::max(col, row);

    offset = cv::Point((maxDimension - col) / 2, (maxDimension - row) / 2);

    cv::Mat result = cv::Mat::zeros(maxDimension, maxDimension, CV_8UC3);
    outputImage.copyTo(result(cv::Rect(offset.x, offset.y, col, row)));
    return result;
}


void detect(const std::string &imagePath, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &classNames) {
    cv::Size originalSize;
    cv::Size readSize;
    cv::Point offset;
    cv::Mat image = preProcessImage(imagePath, originalSize, readSize, offset);
    if (image.empty()) {
        std::cerr << "Couldn't process image: " << imagePath << std::endl;
        return;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    if (outputs.empty()) {
        std::cerr << "Network did not return any outputs." << std::endl;
        return;
    }

    const cv::Mat& outputMat = outputs[0];
    
    // YOLOv8 output format: [1, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
    int numDetections = outputMat.size[2];  // 8400
    int numAttributes = outputMat.size[1];  // 84
    
    float* data = (float*)outputMat.data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Calculate scaling factors from downscaled size to original size
    float x_factor = static_cast<float>(originalSize.width) / readSize.width;
    float y_factor = static_cast<float>(originalSize.height) / readSize.height;
    
    for (int i = 0; i < numDetections; ++i) {
        // YOLOv8 format: [cx, cy, w, h, class0_conf, class1_conf, ..., class79_conf]
        float center_x = data[0 * numDetections + i];
        float center_y = data[1 * numDetections + i];
        float width = data[2 * numDetections + i];
        float height = data[3 * numDetections + i];
        
        // Find the maximum class confidence (no separate objectness score in YOLOv8)
        float max_confidence = 0.0f;
        int best_class_id = 0;
        
        for (int c = 0; c < 11; ++c) {  // 80 COCO classes
            float class_conf = data[(4 + c) * numDetections + i];
            if (class_conf > max_confidence) {
                max_confidence = class_conf;
                best_class_id = c;
            }
        }
        
        // Only process detections above threshold
        if (max_confidence >= CONFIDENCE_THRESHOLD) {
            std::cout << "Detection found: class=" << classNames[best_class_id] 
                      << " (ID=" << best_class_id << ") confidence=" << max_confidence << std::endl;
            
            // Convert from center format to top-left corner format
            float x = center_x - width / 2.0f;
            float y = center_y - height / 2.0f;

            // Remove padding offset to get coordinates on downscaled rectangular image
            float x_unpadded = x - offset.x;
            float y_unpadded = y - offset.y;
            
            // Scale to original image size
            int left = static_cast<int>(x_unpadded * x_factor);
            int top = static_cast<int>(y_unpadded * y_factor);
            int box_width = static_cast<int>(width * x_factor);
            int box_height = static_cast<int>(height * y_factor);

            // Clamp to image bounds
            left = std::max(0, left);
            top = std::max(0, top);
            if (left + box_width > originalSize.width) 
                box_width = originalSize.width - left;
            if (top + box_height > originalSize.height) 
                box_height = originalSize.height - top;

            // Only add valid boxes
            if (box_width > 0 && box_height > 0) {
                class_ids.push_back(best_class_id);
                confidences.push_back(max_confidence);
                boxes.emplace_back(left, top, box_width, box_height);
            }
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_result);

    for (int idx : nms_result) {
        Detection result;
        result.classID = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}


cv::Scalar getClassColor(int classID) {
    // Generate deterministic colors based on class ID
    cv::RNG rng(classID * 12345); // Use class ID as seed for consistent colors
    return cv::Scalar(rng.uniform(100, 255), rng.uniform(100, 255), rng.uniform(100, 255));
}


void drawDetections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& classNames) {
    
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        
        // Draw bounding box
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        
        // Bounds check before accessing classNames
        std::string className;
        if (det.classID >= 0 && det.classID < classNames.size()) {
            className = classNames[det.classID];
        } else {
            className = "Unknown";
            std::cout << "Class name: Unknown (classID " << det.classID << " out of bounds)" << std::endl;
        }
        
        std::string label = className + " " + cv::format("%.2f", det.confidence);
        
        // Get text size
        int baseline = 0;
        float fontScale = 0.8;  // Larger font
        int thickness = 2;      // Thicker text
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
        
        // Calculate label position - ensure it's visible
        int label_x = det.box.x;
        int label_y = det.box.y - 10;
        
        // If label would be above image, place it inside the box
        if (label_y - labelSize.height < 0) {
            label_y = det.box.y + labelSize.height + 20;
        }
        
        // Ensure label doesn't go outside right edge
        if (label_x + labelSize.width > image.cols) {
            label_x = image.cols - labelSize.width - 5;
        }
        
        // Draw label background rectangle for high contrast
        cv::Rect labelBg(label_x - 3, label_y - labelSize.height - 5, 
                        labelSize.width + 6, labelSize.height + 8);
        
        // Ensure background rectangle is within image bounds
        labelBg.x = std::max(0, labelBg.x);
        labelBg.y = std::max(0, labelBg.y);
        if (labelBg.x + labelBg.width > image.cols) labelBg.width = image.cols - labelBg.x;
        if (labelBg.y + labelBg.height > image.rows) labelBg.height = image.rows - labelBg.y;
        
        // Draw bright background
        cv::rectangle(image, labelBg, cv::Scalar(0, 255, 0), cv::FILLED);
        
        // Draw text with black color for maximum contrast
        cv::putText(image, label, cv::Point(label_x, label_y), 
                   cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
    }
}


bool runSingleImageProcessing(const std::string& inputPath, const std::string& outputPath, cv::dnn::Net& net, std::vector<std::string>& classNames) {
    std::vector<Detection> detections;
    detect(inputPath, net, detections, classNames);
    
    // Load original image to display/save results
    cv::Mat originalImage = cv::imread(inputPath);
    if (originalImage.empty()) {
        std::cerr << "Failed to load original image for annotation" << std::endl;
        return false;
    }
    
    // Use the drawDetections function instead of manual drawing
    drawDetections(originalImage, detections, classNames);
    
    if (!cv::imwrite(outputPath, originalImage)) {
        std::cerr << "Failed to save annotated image to: " << outputPath << std::endl;
        return false;
    } else {
        // std::cout << "Annotated image saved to: " << outputPath << std::endl;
        return true;
    }
}