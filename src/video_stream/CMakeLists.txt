cmake_minimum_required(VERSION 3.5.1)
project(videoStream)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g") #-O0 -ggdb

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/include
        ${CUDA_INCLUDE_DIRS}
)

file(GLOB sources *.cpp base64/*cpp)

add_library(videoStream SHARED ${sources})
target_link_libraries(videoStream ${OpenCV_LIBS}
        cuda
        ${CUDA_LIBRARIES}
        )


