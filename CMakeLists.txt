cmake_minimum_required(VERSION 3.10)
project(Rune)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(fivepoints_infer infer/fivepoints_infer.cpp)

target_link_libraries(fivepoints_infer
    openvino::runtime
    ${OpenCV_LIBS}
)

target_include_directories(fivepoints_infer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
