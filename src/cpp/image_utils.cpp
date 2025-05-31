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

std::tuple<std::vector<float>, std::vector<long>, cv::Mat> read_image(const std::string& image_path, int image_size) {
    auto image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Could not read the image: " + image_path);
    }
    if (image.channels() != 3) {
        throw std::runtime_error("Image must have 3 channels (RGB)");
    }

    downscaleImage(image, image, cv::Size(image_size, image_size));

    std::vector<long> input_shape = {1, 3, image_size, image_size};

    // 4) Turn the cv::Mat into a “blob” in NCHW float format, normalized to [0,1]
    cv::Mat nchw = cv::dnn::blobFromImage(
        image,     // the resized BGR image
        1.0f,      // no extra scaling (we’ll divide by 255 ourselves)
        {},        // (we’ve already resized, so no need for a target size here)
        {},        // no mean subtraction
        true       // convert BGR→RGB (the “true” here swaps channels)
    ) / 255.f;    // divide every pixel by 255: 0–255 → 0.0–1.0
    
    // 5) Flatten that blob’s data into a 1D Array<float>
    std::vector<float> input_data(nchw.begin<float>(), nchw.end<float>());

    return {input_data, input_shape, image};
}
