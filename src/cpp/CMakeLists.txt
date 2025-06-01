cmake_minimum_required(VERSION 3.5)

# ----------------------------------------------------------------------------
# 1. Project Name & C++ Standard
# ----------------------------------------------------------------------------
project(GuardianSight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------------
# 2. Find OpenCV
# ----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

# ----------------------------------------------------------------------------
# 3. ONNX Runtime Path
# ----------------------------------------------------------------------------
# Adjust if ORT installed elsewhere
set(ONNX_RUNTIME_PATH "/opt/onnxruntime")

# ----------------------------------------------------------------------------
# 4. Add Executable Target (both .cpp files)
# ----------------------------------------------------------------------------
add_executable(GuardianSight
    main.cpp
    image_utils.cpp
)

# ----------------------------------------------------------------------------
# 5. Include Directories (OpenCV + ONNX Runtime headers)
# ----------------------------------------------------------------------------
target_include_directories(GuardianSight PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    "${ONNX_RUNTIME_PATH}/include"
)

# ----------------------------------------------------------------------------
# 6. Link Directories (where ORT .so files live)
# ----------------------------------------------------------------------------
target_link_directories(GuardianSight PUBLIC
    "${ONNX_RUNTIME_PATH}/lib"
)

# ----------------------------------------------------------------------------
# 7. Link Libraries
#    - OpenCV
#    - onnxruntime (core ORT)
# ----------------------------------------------------------------------------
target_link_libraries(GuardianSight PUBLIC
    ${OpenCV_LIBS}
    onnxruntime
)

# ----------------------------------------------------------------------------
# 8. RPATH so loader finds ORT libs in /opt/onnxruntime/lib at runtime
# ----------------------------------------------------------------------------
set_target_properties(GuardianSight PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${ONNX_RUNTIME_PATH}/lib"
)

# ----------------------------------------------------------------------------
# 9. Extra Compiler Warnings
# ----------------------------------------------------------------------------
target_compile_options(GuardianSight PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)
