cmake_minimum_required(VERSION 3.5)

# ----------------------------------------------------------------------------
# 1. Project Name & C++ Standard
# ----------------------------------------------------------------------------
# Renamed from "YOLO_CXX" to "GuardianSight"
project(GuardianSight LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# 2. Find OpenCV
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# ----------------------------------------------------------------------------
# 3. ONNX Runtime Path
# ----------------------------------------------------------------------------
# Adjust this if ONNXRuntime is installed somewhere else
set(ONNX_RUNTIME_PATH "/opt/onnxruntime")

# ----------------------------------------------------------------------------
# 4. Add Executable Target
# ----------------------------------------------------------------------------
# We compile main.cpp into an executable named "GuardianSight"
add_executable(GuardianSight
    main.cpp
)

# ----------------------------------------------------------------------------
# 5. Include Directories
# ----------------------------------------------------------------------------
#   - OpenCV headers
#   - ONNX Runtime headers
target_include_directories(GuardianSight PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    "${ONNX_RUNTIME_PATH}/include"
)

# ----------------------------------------------------------------------------
# 6. Link Directories
# ----------------------------------------------------------------------------
# Tell CMake where to look for ONNX Runtime libraries
target_link_directories(GuardianSight PUBLIC
    "${ONNX_RUNTIME_PATH}/lib"
)

# ----------------------------------------------------------------------------
# 7. Link Libraries
# ----------------------------------------------------------------------------
#   - OpenCV (as found by find_package)
#   - onnxruntime (the ONNX Runtime shared library)
target_link_libraries(GuardianSight PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
)

target_compile_options(GuardianSight PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
