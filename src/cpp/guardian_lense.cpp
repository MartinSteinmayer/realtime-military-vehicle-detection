// capture_and_process_disk_v2.cpp
//
// - Captures from the Pi camera at ~15 FPS.
// - For each frame, writes it to "/tmp/input.png".
// - Calls preProcessImageOnDisk("/tmp/input.png", "/tmp/output.png").
//     (You should replace this stub with your actual header & function.)
// - Loads "/tmp/output.png" back into OpenCV.
// - Either displays it or writes it to a VideoWriter, based on --mode.
//
// Usage examples:
//   Live display:
//     ./guardian_lense --mode display
//
//   Save to disk:
//     ./guardian_lense --mode save --output processed.avi
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>

#include "image_utils.h"

// --------------------------------------------------------------------------------
// Print usage if arguments are missing or invalid.
// --------------------------------------------------------------------------------
void printUsage(const char* progName)
{
    std::cerr << "Usage:\n"
              << "  " << progName << " --mode <display|save> [--output <out.avi>]\n\n"
              << "  --mode    Either \"display\" or \"save\".\n"
              << "  --output  (Required if --mode save) Path to the output video file.\n\n"
              << "This program will:\n"
              << "  1) Capture a frame from the camera at ~15 FPS.\n"
              << "  2) Write it to \"/tmp/input.png\".\n"
              << "  3) Call preProcessImageOnDisk(\"/tmp/input.png\", \"/tmp/output.png\").\n"
              << "     Replace that stub with your actual header/function.\n"
              << "  4) Read \"/tmp/output.png\" and either display it or save it.\n";
}


int main(int argc, char* argv[]) {
    // -------------------------
    // 1) Parse command‐line args
    // -------------------------
    std::string mode;
    std::string outputVideo;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            outputVideo = argv[++i];
        }
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }
    if (mode != "display" && mode != "save") {
        std::cerr << "Error: --mode must be either \"display\" or \"save\".\n";
        printUsage(argv[0]);
        return 1;
    }
    if (mode == "save" && outputVideo.empty()) {
        std::cerr << "Error: --output <filename> is required when --mode save.\n";
        printUsage(argv[0]);
        return 1;
    }
    // -------------------------------
    // 2) Open the camera (VideoCapture)
    // -------------------------------
    cv::VideoCapture cap(0, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        std::cerr << "ERROR: Cannot open camera (index 0).\n";
        return 1;
    }
    // Try to set it to 15 FPS (some backends ignore this)
    cap.set(cv::CAP_PROP_FPS, 15.0);

    // Grab one frame to determine width & height
    cv::Mat rawFrame;
    cap >> rawFrame;
    if (rawFrame.empty()) {
        std::cerr << "ERROR: Could not grab initial frame.\n";
        return 1;
    }
    int frameW = rawFrame.cols;
    int frameH = rawFrame.rows;
    double actualFps = cap.get(cv::CAP_PROP_FPS);
    if (actualFps <= 0.0) actualFps = 15.0;
    std::cout << "Camera opened at ~" << actualFps << " FPS, "
              << "resolution " << frameW << "×" << frameH << ".\n";

    // -------------------------------------
    // 3) If saving, open a VideoWriter now
    // -------------------------------------
    cv::VideoWriter writer;
    if (mode == "save") {
        int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
        bool ok = writer.open(outputVideo, fourcc, actualFps, cv::Size(frameW, frameH));
        if (!ok) {
            std::cerr << "ERROR: Could not open VideoWriter for \"" << outputVideo << "\".\n";
            return 1;
        }
        std::cout << "Writing processed video to \"" << outputVideo << "\" at " 
                  << actualFps << " FPS.\n";
    }
    else {
        cv::namedWindow("Processed", cv::WINDOW_AUTOSIZE);
        std::cout << "Displaying processed frames in window \"Processed\".\n";
    }

    // Hard‐coded paths for intermediate images
    const std::string inputImagePath  = "/tmp/input.png";
    const std::string outputImagePath = "/tmp/output.png";

    // To keep ~15 FPS even if processing is fast
    const double targetFps = 15.0;
    const auto frameInterval = std::chrono::milliseconds(
        static_cast<int>(1000.0 / targetFps + 0.5)
    );
    auto lastTime = std::chrono::steady_clock::now();

    // Load model
    // std::string modelPath = "yolov8n.onnx";
    std::string modelPath = "best.onnx";
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    if (net.empty()) {
        std::cerr << "Couldn't load net." << std::endl;
        return -1;
    }

    // Load class names
    std::string classFilePath = "class_names.txt";
    std::vector<std::string> classNames = loadClassNames(classFilePath);
    if (classNames.empty()) {
        std::cerr << "Failed to load class names from: " << classFilePath << std::endl;
        return -1;
    }
    std::cout << "Loaded " << classNames.size() << " class names" << std::endl;

    // Main loop
    std::cout << "Starting main loop. Press 'q' in display mode to quit.\n";
    while (true) {
        // 4a) Grab a raw frame
        cap >> rawFrame;
        if (rawFrame.empty()) {
            std::cerr << "WARNING: Empty frame grabbed; exiting loop.\n";
            break;
        }

        // 4b) Throttle to ~15 FPS
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime);
        if (elapsed < frameInterval) {
            std::this_thread::sleep_for(frameInterval - elapsed);
        }
        lastTime = std::chrono::steady_clock::now();

        // 4c) Write raw frame to disk
        if (!cv::imwrite(inputImagePath, rawFrame)) {
            std::cerr << "ERROR: Failed to write \"" << inputImagePath << "\".\n";
            break;
        }

        // Run detection for specified image
        runSingleImageProcessing(inputImagePath, outputImagePath, net, classNames);

        // 4e) Read the processed frame
        cv::Mat procFrame = cv::imread(outputImagePath, cv::IMREAD_COLOR);
        if (procFrame.empty()) {
            std::cerr << "ERROR: Could not read \"" << outputImagePath << "\".\n";
            break;
        }

        // 4f) Display or write
        if (mode == "display") {
            cv::imshow("Processed", procFrame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {  // 'q' or ESC to quit
                std::cout << "Quit key pressed. Exiting loop.\n";
                break;
            }
        }
        else { // save
            writer.write(procFrame);
        }
    }

    // ------------------------------------
    // 5) Cleanup
    // ------------------------------------
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