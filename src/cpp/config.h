#pragma once

#include <string>

class Config {
public:
    Config() = default;
    //loads config struct from a JSON file
    static Config loadConfig(const std::string& filePath);

    // Accessor methods
    const bool verbose() const;
    const std::string& modelPath() const;
    const std::string& classNamesPath() const;
    const std::string& mode() const;
    const std::string& outputPath() const;
    const size_t fpsLimit() const;

private:
    Config(const bool verbose,
            const std::string& modelPath,
            const std::string& classNamesPath,
            const std::string& mode,
            const std::string& outputPath,
            size_t fpsLimit);

    bool _verbose;
    std::string _modelPath;
    std::string _classNamesPath;
    std::string _mode;
    std::string _outputPath;
    size_t _fpsLimit;
};
