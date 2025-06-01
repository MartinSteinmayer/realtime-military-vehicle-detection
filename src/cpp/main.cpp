#include "image_utils.h"
#include "file_utils.h"



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
    
    std::cout << "Loaded " << classNames.size() << " class names" << std::endl;
    
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
    if (originalImage.empty()) {
        std::cerr << "Failed to load original image for annotation" << std::endl;
        return -1;
    }
    
    // Use the drawDetections function instead of manual drawing
    drawDetections(originalImage, detections, classNames);
    
    std::string outputPath = getAnnotatedFilename(imagePath, "./output");
    if (!cv::imwrite(outputPath, originalImage)) {
        std::cerr << "Failed to save annotated image to: " << outputPath << std::endl;
    } else {
        std::cout << "Annotated image saved to: " << outputPath << std::endl;
    }
    
    return 0;
}