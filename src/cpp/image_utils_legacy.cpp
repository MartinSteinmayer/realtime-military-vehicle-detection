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
    // OpenCV 4.6 compatible blob creation - ensure correct channel order
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    try {
        net.forward(outputs, net.getUnconnectedOutLayersNames());
    } catch (const cv::Exception& e) {
        std::cerr << "Error during network forward pass: " << e.what() << std::endl;
        return;
    }

    if (outputs.empty()) {
        std::cerr << "Network did not return any outputs." << std::endl;
        return;
    }

    const cv::Mat& outputMat = outputs[0];
    
    // Comprehensive output tensor information
    std::cout << "Output tensor dims: " << outputMat.dims << std::endl;
    std::cout << "Output tensor type: " << outputMat.type() << " (should be " << CV_32F << ")" << std::endl;
    std::cout << "Output tensor continuous: " << (outputMat.isContinuous() ? "yes" : "no") << std::endl;
    std::cout << "Output tensor size: ";
    for (int i = 0; i < outputMat.dims; i++) {
        std::cout << outputMat.size[i] << " ";
    }
    std::cout << std::endl;
    
    // Ensure the tensor is the expected type
    if (outputMat.type() != CV_32F) {
        std::cerr << "Error: Expected CV_32F tensor, got type " << outputMat.type() << std::endl;
        return;
    }
    
    // Ensure tensor is continuous for safe pointer access
    cv::Mat flatMat;
    if (!outputMat.isContinuous()) {
        flatMat = outputMat.clone();
    } else {
        flatMat = outputMat;
    }
    
    // Handle different tensor formats more safely
    int numDetections, numAttributes;
    bool isTransposed = false;
    
    if (outputMat.dims == 3) {
        if (outputMat.size[0] == 1 && outputMat.size[1] == 84 && outputMat.size[2] == 8400) {
            // Standard YOLOv8: [1, 84, 8400]
            numDetections = outputMat.size[2];
            numAttributes = outputMat.size[1];
            isTransposed = false;
            std::cout << "Using 3D format [1, 84, 8400]" << std::endl;
        } else if (outputMat.size[0] == 1 && outputMat.size[1] == 8400 && outputMat.size[2] == 84) {
            // Alternative YOLOv8: [1, 8400, 84]
            numDetections = outputMat.size[1];
            numAttributes = outputMat.size[2];
            isTransposed = true;
            std::cout << "Using 3D format [1, 8400, 84]" << std::endl;
        } else {
            std::cerr << "Unsupported 3D tensor format: [" << outputMat.size[0] 
                      << ", " << outputMat.size[1] << ", " << outputMat.size[2] << "]" << std::endl;
            return;
        }
    } else if (outputMat.dims == 2) {
        if (outputMat.size[0] == 8400 && outputMat.size[1] == 84) {
            // 2D format: [8400, 84]
            numDetections = outputMat.size[0];
            numAttributes = outputMat.size[1];
            isTransposed = true;
            std::cout << "Using 2D format [8400, 84]" << std::endl;
        } else if (outputMat.size[0] == 84 && outputMat.size[1] == 8400) {
            // 2D format: [84, 8400]
            numDetections = outputMat.size[1];
            numAttributes = outputMat.size[0];
            isTransposed = false;
            std::cout << "Using 2D format [84, 8400]" << std::endl;
        } else {
            std::cerr << "Unsupported 2D tensor format: [" << outputMat.size[0] 
                      << ", " << outputMat.size[1] << "]" << std::endl;
            return;
        }
    } else {
        std::cerr << "Unsupported tensor dimensions: " << outputMat.dims << std::endl;
        return;
    }

    // Validate expected dimensions
    if (numAttributes < 84) {
        std::cerr << "Error: Expected at least 84 attributes (4 bbox + 80 classes), got " << numAttributes << std::endl;
        return;
    }

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Calculate scaling factors from downscaled size to original size
    float x_factor = static_cast<float>(originalSize.width) / readSize.width;
    float y_factor = static_cast<float>(originalSize.height) / readSize.height;
    
    // Use safer tensor access method
    float* data = (float*)flatMat.data;
    int totalElements = flatMat.total();
    
    std::cout << "Processing " << numDetections << " detections with " << numAttributes << " attributes each" << std::endl;
    
    for (int i = 0; i < numDetections; ++i) {
        float center_x, center_y, width, height;
        
        // Calculate indices more safely
        int bbox_indices[4];
        if (outputMat.dims == 3 && !isTransposed) {
            // 3D format [1, 84, 8400]: attributes first, then detections
            for (int j = 0; j < 4; j++) {
                bbox_indices[j] = j * numDetections + i;
            }
        } else {
            // 2D format [8400, 84] or 3D [1, 8400, 84]: detections first, then attributes
            for (int j = 0; j < 4; j++) {
                bbox_indices[j] = i * numAttributes + j;
            }
        }
        
        // Bounds check before accessing
        bool valid_indices = true;
        for (int j = 0; j < 4; j++) {
            if (bbox_indices[j] >= totalElements) {
                std::cerr << "Index out of bounds: " << bbox_indices[j] << " >= " << totalElements << std::endl;
                valid_indices = false;
                break;
            }
        }
        
        if (!valid_indices) continue;
        
        center_x = data[bbox_indices[0]];
        center_y = data[bbox_indices[1]];
        width = data[bbox_indices[2]];
        height = data[bbox_indices[3]];
        
        // Find the maximum class confidence
        float max_confidence = 0.0f;
        int best_class_id = 0;
        
        for (int c = 0; c < 80; ++c) {  // 80 COCO classes
            int class_idx;
            if (outputMat.dims == 3 && !isTransposed) {
                class_idx = (4 + c) * numDetections + i;
            } else {
                class_idx = i * numAttributes + (4 + c);
            }
            
            // Bounds check
            if (class_idx >= totalElements) {
                std::cerr << "Class index out of bounds: " << class_idx << " >= " << totalElements << std::endl;
                break;
            }
            
            float class_conf = data[class_idx];
            if (class_conf > max_confidence) {
                max_confidence = class_conf;
                best_class_id = c;
            }
        }
        
        // Only process detections above threshold
        if (max_confidence >= CONFIDENCE_THRESHOLD) {
            // Bounds check for class_id
            if (best_class_id >= static_cast<int>(classNames.size())) {
                std::cout << "Warning: class_id " << best_class_id << " >= classNames.size() " << classNames.size() << std::endl;
                continue;
            }
            
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

    // Apply Non-Maximum Suppression (OpenCV 4.6 compatible)
    std::vector<int> nms_result;
    try {
        cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, nms_result);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in NMSBoxes: " << e.what() << std::endl;
        return;
    }

    for (size_t idx_i = 0; idx_i < nms_result.size(); ++idx_i) {
        int idx = nms_result[idx_i];
        if (idx >= 0 && idx < static_cast<int>(class_ids.size())) {
            Detection result;
            result.classID = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            output.push_back(result);
        }
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
        if (det.classID >= 0 && det.classID < static_cast<int>(classNames.size())) {
            className = classNames[det.classID];
        } else {
            className = "Unknown";
            std::cout << "Class name: Unknown (classID " << det.classID << " out of bounds)" << std::endl;
        }
        
        // Use sprintf for OpenCV 4.6 compatibility instead of cv::format
        char conf_str[32];
        sprintf(conf_str, "%.2f", det.confidence);
        std::string label = className + " " + std::string(conf_str);
        
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
        return true;
    }
}