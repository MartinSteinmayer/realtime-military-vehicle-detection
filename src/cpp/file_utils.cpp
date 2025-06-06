#include "file_utils.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;


std::string getAnnotatedFilename(const std::string& imagePath, const std::string& outputDir) {
    fs::path inputPath(imagePath);
    std::string stem = inputPath.stem().string();      // "image"
    std::string ext = inputPath.extension().string();  // ".jpg"
    std::string annotatedFilename = stem + "_annotated" + ext;

    fs::path outputPath = fs::path(outputDir) / annotatedFilename;
    return outputPath.string();
}


std::vector<std::string> loadClassNames(const std::string& path) {
    std::vector<std::string> classNames;
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        classNames.push_back(line);
    }
    return classNames;
}