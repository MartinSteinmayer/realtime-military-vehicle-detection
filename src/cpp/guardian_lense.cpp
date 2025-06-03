#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>

#include "image_utils.h"
#include "file_utils.h"
#include "config.h"

// --------------------------------------------------------------------------------
// Print usage if config is missing or invalid.
// --------------------------------------------------------------------------------
void printUsage(const char* progName)
{
    std::cerr << "Usage:\n"
              << "  " << progName << " <config.json>\n\n"
              << "  Example config.json:\n"
              << "  {\n"
              << "    \"verbose\": true,\n"
              << "    \"modelPath\": \"model_data/best.onnx\",\n"
              << "    \"classNamesPath\": \"model_data/class_names.txt\",\n"
              << "    \"mode\": \"display\",\n"
              << "    \"outputPath\": \"output.avi\",\n"
              << "    \"fpsLimit\": 15\n"
              << "  }"
              << "If the config path is not specified, the program will try to read from config.json."
              << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc != 1 && argc != 2) {
        printUsage(argv[0]);
        return 1;
    }

    // Load config
    Config config;
    const std::string configPath = argc == 2 ? argv[1] : "guardian_config.json";
    try {
        config = Config::loadConfig(configPath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << std::endl;
        return 1;
    }

    const bool verbose = config.verbose();

    std::string mode = config.mode();
    if (mode != "display" && mode != "save") {
        std::cerr << "Error: mode must be either \"display\" or \"save\".\n";
        return 1;
    }

    // Open camera
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera.\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_FPS, config.fpsLimit());

    cv::Mat rawFrame;
    cap >> rawFrame;
    if (rawFrame.empty()) {
        std::cerr << "ERROR: Could not grab initial frame.\n";
        return 1;
    }

    int frameW = rawFrame.cols;
    int frameH = rawFrame.rows;
    double actualFps = cap.get(cv::CAP_PROP_FPS);
    if (actualFps <= 0.0) actualFps = config.fpsLimit();
    
    std::cout << "Camera opened at ~" << actualFps << " FPS, "
              << "resolution " << frameW << "×" << frameH << ".\n";

    // VideoWriter setup
    cv::VideoWriter writer;
    if (mode == "save") {
        int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
        if (!writer.open(config.outputPath(), fourcc, actualFps, cv::Size(frameW, frameH))) {
            std::cerr << "ERROR: Could not open output video: " << config.outputPath() << "\n";
            return 1;
        }
        std::cout << "Saving video to " << config.outputPath() << "\n";
    } else {
        cv::namedWindow("Processed", cv::WINDOW_AUTOSIZE);
        std::cout << "Displaying processed frames.\n";
    }

    const std::string inputImagePath  = "/tmp/input.png";
    const std::string outputImagePath = "/tmp/output.png";

    auto frameInterval = std::chrono::milliseconds(
        static_cast<int>(1000.0 / config.fpsLimit() + 0.5)
    );
    auto lastTime = std::chrono::steady_clock::now();

    // Load model
    cv::dnn::Net net = cv::dnn::readNetFromONNX(config.modelPath());
    if (net.empty()) {
        std::cerr << "Couldn't load net from: " << config.modelPath() << std::endl;
        return -1;
    }

    // Load class names
    std::vector<std::string> classNames = loadClassNames(config.classNamesPath());
    if (classNames.empty()) {
        std::cerr << "Failed to load class names from: " << config.classNamesPath() << std::endl;
        return -1;
    }

    if (verbose) std::cout << "Loaded " << classNames.size() << " class names\n";

    size_t iteration = 0;
    std::cout << "Starting main loop. Press 'q' to quit.\n";

    while (true) {
        cap >> rawFrame;
        if (rawFrame.empty()) {
            std::cerr << "WARNING: Empty frame.\n";
            break;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime);
        if (elapsed < frameInterval) {
            std::this_thread::sleep_for(frameInterval - elapsed);
        }
        lastTime = std::chrono::steady_clock::now();

        if (!cv::imwrite(inputImagePath, rawFrame)) {
            std::cerr << "ERROR: Couldn't write input image.\n";
            break;
        }

        auto start = std::chrono::high_resolution_clock::now();
        runSingleImageProcessing(inputImagePath, outputImagePath, net, classNames, verbose);
        auto end = std::chrono::high_resolution_clock::now();

        if (verbose && iteration % 100 == 0) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Single frame processing time: " << ms << " ms\n";
        }

        cv::Mat procFrame = cv::imread(outputImagePath, cv::IMREAD_COLOR);
        if (procFrame.empty()) {
            std::cerr << "ERROR: Couldn't read output image.\n";
            break;
        }

        if (mode == "display") {
            cv::imshow("Processed", procFrame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;
        } else {
            writer.write(procFrame);
        }

        iteration++;
    }

    if (mode == "display") {
        cv::destroyWindow("Processed");
    }
    cap.release();
    if (mode == "save") {
        writer.release();
    }

    std::cout << "Finished.\n";
    return 0;
}
