#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

/*
Downsizes an Image to the specfied targetSize using optimal interpolation for downscaling.
*/
bool downscaleImage(const cv::Mat& input, cv::Mat& output, const cv::Size& targetSize);

/*
Reads in an image from a path and fills in the given output cv::Mat.
*/
bool readImage(const std::string& imagePath, const cv::Size& targetSize, cv::Mat& outputImage);

/*
Reads in an image path, downscales the image and creates an appropriate YOLO input
*/
bool prepareYOLOInput(const std::string& imagePath, const cv::Size& targetSize, std::vector<float>& outputTensor, cv::Mat& downscaledImage);
