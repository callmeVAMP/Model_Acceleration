cmake_minimum_required(VERSION 3.10)
project(sentiment_infer)

set(CMAKE_CXX_STANDARD 17)  # Updated to C++17

# Set the path to LibTorch
set(Torch_DIR "/mnt/combined/home/parveen/varsha/libtorch/share/cmake/Torch")
set(CMAKE_PREFIX_PATH "/mnt/combined/home/parveen/varsha/libtorch")


# Include LibTorch
find_package(Torch REQUIRED)
#include_directories("/mnt/combined/home/parveen/varsha/vcpkg/packages/nlohmann-json_x64-linux/include/nlohmann")

add_executable(sentiment_infer sentiment_infer.cpp)
target_link_libraries(sentiment_infer "${TORCH_LIBRARIES}")
target_include_directories(sentiment_infer PRIVATE 
    "/mnt/combined/home/parveen/varsha/libtorch/include"
    "/mnt/combined/home/parveen/varsha/libtorch/include/torch/csrc/api/include"
    "/mnt/combined/home/parveen/varsha/libtorch/include/c10"
    #"/mnt/combined/home/parveen/varsha/vcpkg/packages/nlohmann-json_x64-linux/include"
    "/mnt/combined/home/parveen/varsha/vcpkg/packages/nlohmann-json_x64-linux/include/"

)
set_property(TARGET sentiment_infer PROPERTY CXX_STANDARD 17)

##################

# project(sentiment_infer)

# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# add_executable(sentiment_infer sentiment_infer.cpp)
# target_link_libraries(sentiment_infer "${TORCH_LIBRARIES}")
# set_property(TARGET sentiment_infer PROPERTY CXX_STANDARD 14)

# list(APPEND CMAKE_PREFIX_PATH "libtorch")