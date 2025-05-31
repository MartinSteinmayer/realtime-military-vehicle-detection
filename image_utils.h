#include <iostream>
#include <opencv2/opencv.hpp>

/*
Downsizes an Image to the specfied targetSize using optimal interpolation for downscaling.
*/
bool downScaleImage(const cv::Mat& input, cv::Mat& output, const cv::Size& targetSize);




