#include "image_utils.h"


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <onnx_model_path> <class_names.txt>" << std::endl;
        return -1;
    }

    std::string imagePath = argv[1];
    std::string modelPath = argv[2];
    std::string classFilePath = argv[3];

    // Load class names
    std::vector<std::string> classNames = loadClassNames(classFilePath);
    if (classNames.empty()) {
        std::cerr << "Failed to load class names from: " << classFilePath << std::endl;
        return -1;
    }

    // Load the ONNX model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);
    if (net.empty()) {
        std::cerr << "Failed to load the model: " << modelPath << std::endl;
        return -1;
    }

    std::vector<Detection> detections;
    detect(imagePath, net, detections, classNames);

    // Load original image to display/save results
    cv::Mat originalImage = cv::imread(imagePath);
    for (const auto& det : detections) {
        cv::rectangle(originalImage, det.box, cv::Scalar(0, 255, 0), 2);
        std::string label = classNames[det.classID] + " " + cv::format("%.2f", det.confidence);
        cv::putText(originalImage, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    cv::imshow("YOLOv8 Detection", originalImage);
    cv::waitKey(0);  // Wait for a key press
    return 0;
}

