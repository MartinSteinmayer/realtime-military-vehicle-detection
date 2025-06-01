cmake_minimum_required(VERSION 3.10)
project(YOLOv8_Detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories (same as before)
include_directories(${OpenCV_INCLUDE_DIRS} src/cpp)

# OpenCV version detection and conditional source selection
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Determine which image_utils source to use based on OpenCV version
if(OpenCV_VERSION VERSION_GREATER_EQUAL "4.7.0")
    # Modern OpenCV version (4.7+)
    set(IMAGE_UTILS_SOURCE src/cpp/image_utils.cpp)
    add_definitions(-DOPENCV_NEW_VERSION)
    message(STATUS "Using modern OpenCV implementation (image_utils.cpp)")
else()
    # Legacy OpenCV version (4.6 and below)
    set(IMAGE_UTILS_SOURCE src/cpp/image_utils_legacy.cpp)
    add_definitions(-DOPENCV_LEGACY_VERSION)
    message(STATUS "Using legacy OpenCV implementation (image_utils_legacy.cpp)")
endif()

# Optional: Platform detection for Raspberry Pi specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    add_definitions(-DRASPBERRY_PI_BUILD)
    message(STATUS "Building for ARM64/Raspberry Pi")
endif()

# Create executable (same structure as before, but with conditional image_utils)
add_executable(guardian_lense
    src/cpp/guardian_lense.cpp
    ${IMAGE_UTILS_SOURCE}
    src/cpp/file_utils.cpp
)

# Link libraries (same as before)
target_link_libraries(guardian_lense ${OpenCV_LIBS})

# Optional: Manual override via command line
# Usage: cmake -DFORCE_OPENCV_VERSION=legacy ..
#        cmake -DFORCE_OPENCV_VERSION=new ..
if(DEFINED FORCE_OPENCV_VERSION)
    if(FORCE_OPENCV_VERSION STREQUAL "legacy")
        set(IMAGE_UTILS_SOURCE src/cpp/image_utils_legacy.cpp)
        add_definitions(-DOPENCV_LEGACY_VERSION)
        message(STATUS "Forced to use legacy OpenCV implementation")
    elseif(FORCE_OPENCV_VERSION STREQUAL "new")
        set(IMAGE_UTILS_SOURCE src/cpp/image_utils.cpp)
        add_definitions(-DOPENCV_NEW_VERSION)
        message(STATUS "Forced to use modern OpenCV implementation")
    endif()
endif()

# Print final build configuration
message(STATUS "Build configuration:")
message(STATUS "  Project: ${PROJECT_NAME}")
message(STATUS "  Target: guardian_lense")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Image utils source: ${IMAGE_UTILS_SOURCE}")
message(STATUS "  System processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "  Include directories: ${OpenCV_INCLUDE_DIRS} src/cpp")