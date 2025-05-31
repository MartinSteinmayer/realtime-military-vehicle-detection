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


bool readImage(const std::string& imagePath, const cv::Size& targetSize, cv::Mat& outputImage) {
    auto image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to read image: " << imagePath;
        return false;
    }
    if (image.channels() != 3) {
        std::cerr << "Image must have three channels (RBG/BGR)." << std::endl;
        return false;
    }
    return true;
}


bool prepareYOLOInput(const std::string& imagePath, const cv::Size& targetSize, std::vector<float>& outputTensor, cv::Mat& downscaledImage) {
    cv::Mat originalImage;
    if (!readImage(imagePath, targetSize, originalImage)) {
        return false;
    }
    if (!downscaleImage(originalImage, downscaledImage, targetSize)) {
        std::cerr << "Failed to downscale Image." << std::endl;
        return false;
    }

    cv::Mat blob = cv::dnn::blobFromImage(
        downscaledImage,
        1.0f / 255.0f,  // scale factor (normalize 0-255 to 0-1)
        {},             // size: no resizing here, already done
        {},             // mean subtraction: none
        true            // swapRB: BGR to RGB conversion
    );

    // Flatten blob to std::vector<float>
    outputTensor.assign((float*)blob.datastart, (float*)blob.dataend);

    return true;

}
