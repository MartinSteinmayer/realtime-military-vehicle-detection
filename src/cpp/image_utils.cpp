#include "image_utils.h"

bool downscaleImage(const cv::Mat& input, cv::Mat& output, const cv::Size& targetSize){
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
    }
    catch (const cv::Exception &e) {
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
        std::cerr << "Image must have three channels (RBG/BGR)." << std::endl;
        return false;
    }
    return true;
}


cv::Mat prepareYOLOInput(const cv::Mat& src) {
    int col = src.cols;
    int row = src.rows;
    int maxDimension = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(maxDimension, maxDimension, CV_8UC3);
    src.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}


cv::Mat processImage(const std::string& imagePath) {
    cv::Mat inputImage;
    cv::Mat outputImage;
    if (!readImage(imagePath, inputImage)) {
        std::cerr << "No image read." << std::endl;
        return outputImage;
    }
    if (!downscaleImage(inputImage, outputImage, cv::Size(INPUT_WIDTH, INPUT_HEIGHT))) {
        std::cerr << "Image couldn't be downscaled." << std::endl;
        return outputImage;
    }
    return prepareYOLOInput(outputImage);
}


void detect(const std::string &imagePath, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &classNames) {
    
    cv::Mat image = processImage(imagePath);
    if (image.empty()) {
        std::cerr << "Couldn't process image: " << imagePath << std::endl;
        return;
    }

    // Create blob from image and feed to network
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    if (outputs.empty()) {
        std::cerr << "Network did not return any outputs." << std::endl;
        return;
    }

    // YOLOv8 outputs: [x, y, w, h, obj_conf, class_scores...]
    const cv::Mat& outputMat = outputs[0];
    const int numDetections = outputMat.size[1];
    const int numAttributes = outputMat.size[2]; // Usually 85 for COCO: 4 + 1 + 80

    float* data = (float*)outputMat.data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = static_cast<float>(image.cols) / INPUT_WIDTH;
    float y_factor = static_cast<float>(image.rows) / INPUT_HEIGHT;

    for (int i = 0; i < numDetections; ++i) {
        float obj_conf = data[4];

        if (obj_conf >= CONFIDENCE_THRESHOLD) {
            cv::Mat scores(1, classNames.size(), CV_32FC1, data + 5);
            cv::Point classIdPoint;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);

            float confidence = obj_conf * static_cast<float>(max_class_score);
            if (confidence > SCORE_THRESHOLD) {
                int class_id = classIdPoint.x;

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = static_cast<int>((x - 0.5f * w) * x_factor);
                int top = static_cast<int>((y - 0.5f * h) * y_factor);
                int width = static_cast<int>(w * x_factor);
                int height = static_cast<int>(h * y_factor);

                class_ids.push_back(class_id);
                confidences.push_back(confidence);
                boxes.emplace_back(left, top, width, height);
            }
        }

        data += numAttributes;
    }

    // Apply Non-Maximum Suppression
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);

    for (int idx : nms_result) {
        Detection result;
        result.classID = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);

        // Draw results
        cv::rectangle(image, boxes[idx], cv::Scalar(0, 255, 0), 2);
        std::string label = classNames[class_ids[idx]] + " " + cv::format("%.2f", confidences[idx]);
        cv::putText(image, label, cv::Point(boxes[idx].x, boxes[idx].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // Optionally show or save image
    // cv::imshow("Detections", image);
    // cv::waitKey(0);
    // cv::imwrite("output.jpg", image);
}
