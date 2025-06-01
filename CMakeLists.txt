cmake_minimum_required(VERSION 3.10)
project(YOLOv8_Detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} src/cpp)

add_executable(guardian_lense
    src/cpp/guardian_lense.cpp
    src/cpp/image_utils.cpp
    src/cpp/file_utils.cpp
)

target_link_libraries(guardian_lense ${OpenCV_LIBS})