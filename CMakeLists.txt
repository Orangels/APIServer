cmake_minimum_required(VERSION 3.5)
project(DHP)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g -fno-inline")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
# find_package(PythonLibs)  # link to python 3.6  error
# message(${PYTHON_INCLUDE_DIRS})
# message(${PYTHON_LIBRARIES})

set(cppCommonPath ${CMAKE_SOURCE_DIR}/src/cppCommon)
set(videoStreamPath ${CMAKE_SOURCE_DIR}/src/video_stream)
execute_process(COMMAND mkdir imgs)
execute_process(COMMAND mkdir imgs_center)
execute_process(COMMAND mkdir imgs_ori)
execute_process(COMMAND mkdir imgs_detections)

add_subdirectory(${videoStreamPath})

message(${OpenCV_INCLUDE_DIRS})

include_directories(
        include
        ${cppCommonPath}/include
        ${OpenCV_INCLUDE_DIRS}
#        ${PYTHON_INCLUDE_DIRS}
        /usr/include/python3.6m
        /usr/local/lib/python3.6/dist-packages/numpy/core/include
        ${CMAKE_SOURCE_DIR}/src/video_stream
        /usr/local/include
        ${CUDA_INCLUDE_DIRS}
        src/box_tracking
        ${CMAKE_SOURCE_DIR}/pycode

)
link_libraries(
        /usr/lib/aarch64-linux-gnu/libpython3.6m.so
        videoStream
)
aux_source_directory(src DIR_SRC)
message(${DIR_SRC})
add_executable(DPH tools/main.cpp ${DIR_SRC}
#        pycode/EnginePy.cpp
src/utils/mat2numpy.cpp src/utils/vis.cpp src/utils/split.cpp src/utils/match_id.cpp src/utils/trajectory.cpp
include/structures/structs.h
src/box_tracking/box_tracking.cpp
src/box_tracking/box_tracking.h src/box_tracking/Hungarian/Hungarian.cpp
src/utils/track.cpp ${cppCommonPath}/build/libCommon.so)

