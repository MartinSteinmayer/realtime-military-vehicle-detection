#include <filesystem>
#include <fstream>
#include <vector>

/*
Returns the output path of the filename, given the output directory. Will append "_annotated" to the original filename before the extension.
*/
std::string getAnnotatedFilename(const std::string& imagePath, const std::string& outputDir);