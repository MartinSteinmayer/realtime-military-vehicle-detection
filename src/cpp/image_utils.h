#pragma once

#include <string>
#include <opencv2/opencv.hpp>

const int INPUT_WIDTH = 320;
const int INPUT_HEIGHT = 320;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.6;

struct Detection {
    int classID;
    float confidence;
    cv::Rect box;
};


/*
Downsizes an Image to the specfied targetSize using optimal interpolation for downscaling.
*/
bool downscaleImage(const cv::Mat& input, cv::Mat& output, const cv::Size& targetSize);


/*
Reads in an image from a path and fills in the given output cv::Mat.
*/
bool readImage(const std::string& imagePath, cv::Mat& outputImage);


/*
Takes in image path and returns processed, YOLO-conform cv::Mat. Writes the size of the original image in the readSize object.
*/
cv::Mat preProcessImage(const std::string& imagePath, cv::Size& originalSize, cv::Size& readSize, cv::Point& offset);


/*
Performs class detection and annotates the given image
*/
void detect(const std::string &imagePath, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className, const bool verbose = false);


/*
Helper function to get class color
*/
cv::Scalar getClassColor(int classID);


/*
Draw the detected boxes
*/
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& classNames);


/*
Run the full image processing on one image
*/
bool runSingleImageProcessing(const std::string& inputPath, const std::string& outputPath, cv::dnn::Net& net, const std::vector<std::string>& classNames, const bool verbose = false);