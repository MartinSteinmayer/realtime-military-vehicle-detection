#include <iostream>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

const int INPUT_WIDTH = 320;
const int INPUT_HEIGHT = 320;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.5;
const float CONFIDENCE_THRESHOLD = 0.5;

struct Detection {
    int classID;
    float confidence;
    cv::Rect box;
};


/*
Reads in a vector of class names from a file
*/
std::vector<std::string> loadClassNames(const std::string& path);


/*
Downsizes an Image to the specfied targetSize using optimal interpolation for downscaling.
*/
bool downscaleImage(const cv::Mat& input, cv::Mat& output, const cv::Size& targetSize);


/*
Reads in an image from a path and fills in the given output cv::Mat.
*/
bool readImage(const std::string& imagePath, cv::Mat& outputImage);


/*
Takes in a raw cv::Mat and co;nverts it to the expected YOLO input format
*/
cv::Mat prepareYOLOInput(const cv::Mat& src);


/*
Takes in image path and returns processed, YOLO-conform cv::Mat. Writes the size of the original image in the readSize object.
*/
cv::Mat processImage(const std::string& imagePath, cv::Size& readSize);


/*
Performs class detection and annotates the given image
*/
void detect(const std::string &imagePath, cv::dnn::Net &net, std::vector<Detection> &output, const std::vector<std::string> &className);


/*

*/
void drawDetections(cv::Mat& image, const std::vector<Detection>& detections, const std::vector<std::string>& classNames);