#include "config.h"
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

Config::Config(const bool verbose,
               const std::string& modelPath,
               const std::string& classNamesPath,
               const std::string& mode,
               const std::string& outputPath,
               const size_t fpsLimit)
    : _verbose(verbose),
      _modelPath(modelPath),
      _classNamesPath(classNamesPath),
      _mode(mode),
      _outputPath(outputPath),
      _fpsLimit(fpsLimit) {}

Config Config::loadConfig(const std::string& filePath) {
    std::ifstream inFile(filePath);
    if (!inFile.is_open()) {
        throw std::runtime_error("Unable to open config file: " + filePath);
    }
    
    json j;
    inFile >> j;
    
    return Config(
        j.at("verbose").get<bool>(),
        j.at("modelPath").get<std::string>(),
        j.at("classNamesPath").get<std::string>(),
        j.at("mode").get<std::string>(),
        j.at("outputPath").get<std::string>(),
        j.at("fpsLimit").get<size_t>()
    );
}

const bool Config::verbose() const {
    return _verbose;
}

const std::string& Config::modelPath() const { 
    return _modelPath; 
}

const std::string& Config::classNamesPath() const { 
    return _classNamesPath; 
}

const std::string& Config::mode() const { 
    return _mode; 
}

const std::string& Config::outputPath() const { 
    return _outputPath; 
}

const size_t Config::fpsLimit() const { 
    return _fpsLimit; 
}