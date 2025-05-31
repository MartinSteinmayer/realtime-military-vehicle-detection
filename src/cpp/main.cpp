#include "image_utils.h"
#include <onnxruntime_cxx_api.h>

const char* class_names[] = {
    "Person"
};


std::pair<std::vector<float>, std::vector<long>> process_image(Ort::Session &session, std::vector<float> &array, std::vector<long> shape)
{
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto input = Ort::Value::CreateTensor<float>(
        memory_info, (float *)array.data(), array.size(), shape.data(), shape.size());

    const char *input_names[] = {"images"};
    const char *output_names[] = {"output"};
    auto output = session.Run({}, input_names, &input, 1, output_names, 1);
    shape = output[0].GetTensorTypeAndShapeInfo().GetShape();
    auto ptr = output[0].GetTensorData<float>();
    return {std::vector<float>(ptr, ptr + shape[0] * shape[1]), shape};
}

void display_image(cv::Mat image, const std::vector<float> &output, const std::vector<long> &shape)
{
    for (size_t i = 0; i < shape[0]; ++i)
    {
        auto ptr = output.data() + i * shape[1];
        int x = ptr[1], y = ptr[2], w = ptr[3] - x, h = ptr[4] - y, c = ptr[5];
        auto color = CV_RGB(255, 255, 255);
        auto name = std::string(class_names[c]) + ":" + std::to_string(int(ptr[6] * 100)) + "%";
        cv::rectangle(image, {x, y, w, h}, color);
        cv::putText(image, name, {x, y}, cv::FONT_HERSHEY_DUPLEX, 1, color);
    }

    cv::imshow("Yolov8n Output", image);
    cv::waitKey(0);
}

int main() {
    bool use_cuda = false;
    int image_size = 320;
    std::string model_path = "model.onnx";
    std::string image_path = "image.jpg";
    /*const char* class_names[] = {*/
    /*    "Tank",*/
    /*    "Anti-aircraft vehicle",*/
    /*    "Armored combat support vehicle",*/
    /*    "Infantry fighting vehicle",*/
    /*    "Armored personnel carrier",*/
    /*    "Self-propelled artillery",*/
    /*    "Mine-protected vehicle",*/
    /*    "Truck",*/
    /*    "Light armored vehicle",*/
    /*    "Light utility vehicle",    */
    /*};*/


    std::tuple<std::vector<float>, std::vector<long>, cv::Mat> result = read_image(image_path, image_size);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Yolov8n");
    Ort::SessionOptions options;
    if (use_cuda) Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(options, 0));
    Ort::Session session(env, model_path.c_str(), options);


    auto input_data = std::get<0>(result);
    auto input_shape = std::get<1>(result);
    cv::Mat image = std::get<2>(result);


    auto output = process_image(session, input_data, input_shape);

    display_image(image, output.first, output.second);
}
