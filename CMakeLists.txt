cmake_minimum_required(VERSION 3.5)
project(APIServer)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g -fno-inline")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
# find_package(PythonLibs)  # link to python 3.6  error
# message(${PYTHON_INCLUDE_DIRS})
# message(${PYTHON_LIBRARIES})

#wiwide tensorrt
set(protoDIR ${CMAKE_SOURCE_DIR}/onnx)
set(wiwide_tr ${CMAKE_SOURCE_DIR}/src/tensorrtFrame)
execute_process(COMMAND  protoc  --cpp_out=${protoDIR}  -I=${protoDIR} ${protoDIR}/onnx.proto)
execute_process(COMMAND  protoc  --cpp_out=${CMAKE_SOURCE_DIR}  -I=${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/caffe.proto)
execute_process(COMMAND  ln -s ${wiwide_tr}/functions4cpp.py ${wiwide_tr}/../../../functions4cpp.py)
execute_process(COMMAND  ln -s ${wiwide_tr}/functions4cpp.py ${CMAKE_SOURCE_DIR}/build/functions4cpp.py)

find_package(Protobuf REQUIRED)
execute_process(COMMAND chmod +x ${wiwide_tr}/createDetectionTool.sh )
execute_process(COMMAND sh ${wiwide_tr}/createDetectionTool.sh  ${wiwide_tr}/convertProtobufBinary2txt)


set(cppCommonPath ${CMAKE_SOURCE_DIR}/src/cppCommon)
set(videoStreamPath ${CMAKE_SOURCE_DIR}/src/video_stream)
execute_process(COMMAND mkdir imgs)
execute_process(COMMAND mkdir imgs_center)
execute_process(COMMAND mkdir imgs_ori)
execute_process(COMMAND mkdir imgs_detections)


if ("${CMAKE_SYSTEM}" MATCHES ".*tegra")
    add_definitions(-DPLATFORM_TEGRA)
    set(trtLibPath /usr/lib/aarch64-linux-gnu)
    set(cppComLib ${cppCommonPath}/build/libCommon.so)
#    execute_process(COMMAND  ln -s /srv/models ${CMAKE_SOURCE_DIR}/../../../models)
else()
    set(cppComLib ${cppCommonPath}/build/libCommon.so)
    execute_process(COMMAND  ln -s /home1/VisionProjectModel/caffe ${CMAKE_SOURCE_DIR}/../../../models)
    set(trtLibPath /usr/lib/x86_64-linux-gnu)
    set(CUDA_NVCC_FLAGS
            ${CUDA_NVCC_FLAGS}
            -O3 #-g -G  # -O3 #
            -gencode arch=compute_30,code=sm_30
            -gencode arch=compute_35,code=sm_35
            -gencode arch=compute_50,code=sm_50
            -gencode arch=compute_52,code=sm_52
            -gencode arch=compute_61,code=sm_61
            -gencode arch=compute_62,code=sm_62
            -gencode arch=compute_70,code=sm_70
            )
endif()


add_subdirectory(${videoStreamPath})

message(${OpenCV_INCLUDE_DIRS})

include_directories(
        include
        ${wiwide_tr}
        ${cppCommonPath}/include
        ${OpenCV_INCLUDE_DIRS}
        ${PROTOBUF_INCLUDE_DIR}
#        ${PYTHON_INCLUDE_DIRS}
        /usr/include/python3.6m
        /usr/local/lib/python3.6/dist-packages/numpy/core/include
        ${CMAKE_SOURCE_DIR}/src/video_stream
        ${CMAKE_SOURCE_DIR}/src/video_stream/DSIncludes
        /usr/local/include
        ${CUDA_INCLUDE_DIRS}
        src/box_tracking
        ${CMAKE_SOURCE_DIR}/pycode

        /usr/include/gstreamer-1.0
        /usr/include/glib-2.0
        /usr/lib/aarch64-linux-gnu/glib-2.0/include
        /usr/local/cuda/targets/aarch64-linux/include/
        /usr/local/ffmpeg/include

)
aux_source_directory(src DIR_SRC)
message(${DIR_SRC})

file(GLOB sources ${wiwide_tr}/*.cpp ${wiwide_tr}/*.cc ${wiwide_tr}/onnx/*.cc)
file(GLOB cudaSources ${wiwide_tr}/*.cu)

cuda_add_library(inferLib STATIC ${cudaSources})

link_libraries(
        /usr/lib/aarch64-linux-gnu/libpython3.6m.so
        videoStream
        ${PROTOBUF_LIBRARIES}
        inferLib nvinfer nvcaffeparser  nvinfer_plugin nvparsers ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${trtLibPath}/libcudnn.so ${cppComLib}
)

add_executable(APIServer tools/main.cpp ${DIR_SRC} ${sources}
        pycode/EnginePy.cpp
src/utils/mat2numpy.cpp src/utils/vis.cpp src/utils/split.cpp src/utils/match_id.cpp src/utils/trajectory.cpp
include/structures/structs.h
src/box_tracking/box_tracking.cpp
src/box_tracking/box_tracking.h src/box_tracking/Hungarian/Hungarian.cpp src/tasks/imageHandler.cpp
src/utils/track.cpp ${cppCommonPath}/build/libCommon.so)

