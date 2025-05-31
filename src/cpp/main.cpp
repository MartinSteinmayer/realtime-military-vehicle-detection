#include "image_utils.h"

int main() {
    bool use_cuda = false;
    int image_size = 320;
    std::string model_path = "model.onnx";
    std::string image_path = "image.jpg";
    const char* class_names[] = {
        "Tank",
        "Anti-aircraft vehicle",
        "Armored combat support vehicle",
        "Infantry fighting vehicle",
        "Armored personnel carrier",
        "Self-propelled artillery",
        "Mine-protected vehicle",
        "Truck",
        "Light armored vehicle",
        "Light utility vehicle",    
    };
}