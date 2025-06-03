# GuardianSight
## A lightweight solution for real-time military vehicle detection

## Background
This project started being developed for the [European Defense Tech Hub](https://eurodefense.tech/) challenge in Drone Vision.
The purpose was to deliver a lightweight program for real-time object detection that could run fully autonomously in a drone's hardware.
To demonstrate this, we extensively tested the program on a **Raspberry Pi 5** running *Ubuntu Server*.
As of now, the program can use any YOLOv8n-compatible model (given a list of desired class names) and perform real-time visual classification.

## Setup & Installation
### Python utilities
If one needs to create their own model they can use the provided python functionality to fine-tune a YOLOv8 model to a set of images.
It is recommended to create a virtual environment using:
```
python3 -m venv <venv-name>
```
And subsequently installing the necessary dependencies:
```
pip install -r "requirements.txt"
```
Please refer to your system's Python documentation for activating the virtual environment. [Here](https://docs.python.org/3/library/venv.html) is a useful reference.

### Dependencies & Compilation
To compile the program, one needs to install [OpenCV](https://opencv.org/) (>4.8) and [nlohmann/json](https://github.com/nlohmann/json).
For the compilation, it is easiest to use [CMake](https://cmake.org/) and follow these steps:
1. Starting from this repository's root, create a **build** directory:
```
mkdir build
```
2. Enter the **build** directory:
```
cd build/
```
3. Generate build files:
```
cmake ..
```
4. Compile:
```
make
```
5. Move the **generated executable** to the desired location, e.g.:
```
mv guardian_sight ~/Desktop
```

## Usage
### Python
Under `src/python/` you will find *most* of the code that we used during the project to label the datasets using the large GroundingDINO model and subsequently fine-tune the YOLO model used by the GuardianSight program. Please take a look at the provided scripts for the usage.

### GuardianSight
To run the program, one needs to create a `config.json` file containing the parameters for the program to run. A template is provided under `guardian_config.json`.
This is also set as the **default** config file name if the config file path is not specified when the program is called. These are the config parameters (in the expected format):
```
{
    "verbose": false,                       # Print additional information such as detections with confidence, runtime processing single images, etc.
    "modelPath": "yolov8n.onnx",            # Model used by the program
    "classNamesPath": "class_names.txt",    # Text file containing the class names in order
    "mode": "display",                      # Option to choose between real-time display of detections or saving the recording
    "outputPath": "library_test.avi",       # If save mode is on, output path where recording will be saved
    "fpsLimit": 15                          # FPS limit for the recording/display
}
```
>Note: The comments inside the JSON example are for explanation purposes only and are not valid JSON syntax. When creating your actual config.json file, please remove all comments.

The program relies on a `.onnx` model to run with OpenCV, and the path must be specified in the config. As stated, the provided model must be **compatible with the YOLOv8n base model**.
Two template models (the ones we used during the challenge) are provided under `model_data/`. Another important parameter is the **ordering of class names** - this should be provided as a `.txt` file and the path specified in the configuration. The templates under `model_data/` are, once again, the ones we used during the Hackathon.
>Note: Since most YOLO models **do not** save an additional encoding of *output layer index* to *class name*, the model will generally assing the predicted class an index, not an actual string. Therefore, it is necessary to provide the desired encoding **in the same order** as the one used to train the model.

To **start GuardianSight**, simply run:
```
./guardian_sight <config_path>
```
And watch as the program swiftly classifies objects that appear on camera. To **exit** the program, simply press `q`.
>Note: Take a look at the `mjpeg_streamer.py` script for real-time streaming to the localhost.

