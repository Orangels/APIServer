cmake_minimum_required(VERSION 3.5.1)
project(tensorrtFrame)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -fPIC ") #-O0 -ggdb
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -ggdb -Wall -fPIC ") #-O0 -ggdb

find_package(CUDA) 
if(9 LESS ${CUDA_VERSION_MAJOR}) 
   execute_process(COMMAND rm ${CMAKE_SOURCE_DIR}/float32halfConvertor.cpp)
endif()

#set(OpenCV_DIR G:/opencv32/build/install)
find_package(OpenCV REQUIRED)
set(protoDIR ${CMAKE_SOURCE_DIR}/onnx)
execute_process(COMMAND  protoc  --cpp_out=${protoDIR}  -I=${protoDIR} ${protoDIR}/onnx.proto)
execute_process(COMMAND  protoc  --cpp_out=${CMAKE_SOURCE_DIR}  -I=${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/caffe.proto)
execute_process(COMMAND  ln -s ${CMAKE_SOURCE_DIR}/functions4cpp.py ${CMAKE_SOURCE_DIR}/../../../functions4cpp.py)
execute_process(COMMAND  ln -s ${CMAKE_SOURCE_DIR}/functions4cpp.py ${CMAKE_SOURCE_DIR}/build/functions4cpp.py)

find_package(Protobuf REQUIRED)
execute_process(COMMAND chmod +x ${CMAKE_SOURCE_DIR}/createDetectionTool.sh )
execute_process(COMMAND sh ${CMAKE_SOURCE_DIR}/createDetectionTool.sh  ${CMAKE_SOURCE_DIR}/convertProtobufBinary2txt)
set(cppCommonPath ${CMAKE_SOURCE_DIR}/../cppCommon)
include_directories(/usr/include ${OpenCV_INCLUDE_DIRS} ${cppCommonPath}/include ${PROTOBUF_INCLUDE_DIR} )#${PYTHON_INCLUDE_DIRS}

execute_process(COMMAND  rm ${CMAKE_SOURCE_DIR}/model)
execute_process(COMMAND  rm ${CMAKE_SOURCE_DIR}/../../../models)
if ("${CMAKE_SYSTEM}" MATCHES ".*tegra")
    set(trtLibPath /usr/lib/aarch64-linux-gnu)
    set(cppComLib ${CMAKE_SOURCE_DIR}/../cppCommon/build/libCommon.so)
	execute_process(COMMAND  ln -s /srv/models ${CMAKE_SOURCE_DIR}/../../../models)
else()
	set(cppComLib ${CMAKE_SOURCE_DIR}/../cppCommon/build/libCommon.so)
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
execute_process(COMMAND  ln -s ${CMAKE_SOURCE_DIR}/../../../models ${CMAKE_SOURCE_DIR}/model)

file(GLOB sources *.cpp *.cc onnx/*.cc)
file(GLOB cudaSources *.cu)

cuda_add_library(inferLib STATIC ${cudaSources})
# add_library(tensorrtFrame SHARED ${sources})
add_executable(tensorrtFrame ${sources})
target_link_libraries(tensorrtFrame  ${PROTOBUF_LIBRARIES} inferLib nvinfer nvcaffeparser  nvinfer_plugin nvparsers ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${trtLibPath}/libcudnn.so ${cppComLib} )

                                        
