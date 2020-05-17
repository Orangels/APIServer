cmake_minimum_required(VERSION 3.5.1)
project(videoStream)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g") #-O0 -ggdb

find_package(OpenCV REQUIRED)
find_package(GFLAGS REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)

if (GLEW_FOUND)
    message(found glew)
    message(${GLEW_INCLUDE_DIRS})
    include_directories(${GLEW_INCLUDE_DIRS})
    link_libraries(${GLEW_LIBRARIES})
endif ()

message(${PROJECT_SOURCE_DIR}/IMV/include)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/ffmpeg/include
        ${PROJECT_SOURCE_DIR}/DSIncludes
        /usr/local/include
        /usr/include/gstreamer-1.0
        /usr/include/glib-2.0
        ${gflags_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/IMV/include
)

file(GLOB sources *.cpp base64/*cpp fishCamHandler/*cpp)

add_library(videoStream SHARED ${sources})
target_link_libraries(videoStream ${OpenCV_LIBS}
        /opt/nvidia/deepstream/deepstream-4.0/lib/libnvdsgst_meta.so
        /opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_meta.so
        /opt/nvidia/deepstream/deepstream-4.0/lib/libnvbufsurftransform.so
        /opt/nvidia/deepstream/deepstream-4.0/lib/libnvbufsurface.so

        /usr/local/lib/libgearman.so
        /usr/local/ffmpeg/lib/libavcodec.so
        /usr/local/ffmpeg/lib/libavdevice.so
        /usr/local/ffmpeg/lib/libavfilter.so
        /usr/local/ffmpeg/lib/libavformat.so
        /usr/local/ffmpeg/lib/libavutil.so
        /usr/local/ffmpeg/lib/libpostproc.so
        /usr/local/ffmpeg/lib/libswresample.so
        /usr/local/ffmpeg/lib/libswscale.so
        ${PROJECT_SOURCE_DIR}/IMV/lib/libIMV1.so
        gflags
        GLEW glfw3 GL
        X11 Xrandr Xxf86vm Xinerama Xcursor rt m dl
        cuda
        ${CUDA_LIBRARIES}
        )


