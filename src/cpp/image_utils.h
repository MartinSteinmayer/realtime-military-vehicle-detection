#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

/*
Downsizes an Image to the specfied targetSize using optimal interpolation for downscaling.
*/
bool downscaleImage(const cv::Mat& input, cv::Mat& output, const cv::Size& targetSize);

std::tuple<std::vector<float>, std::vector<long>, cv::Mat> read_image(const std::string& image_path, int image_size);
