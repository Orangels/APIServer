cmake_minimum_required(VERSION 3.5.1)
project(postPara3DKeyPointsProtobufferConvertor)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -fPIC") #-O0 -ggdb

find_package(Protobuf REQUIRED)
set(protoDIR ${CMAKE_SOURCE_DIR})
include_directories(/usr/include/ ${PROTOBUF_INCLUDE_DIR})
message(-------------${PROTOBUF_INCLUDE_DIR})

file(GLOB sources main.cpp postPara3DKeyPoints.pb.cc)
add_executable(postPara3DKeyPointsProtobufferConvertor ${sources})
target_link_libraries(postPara3DKeyPointsProtobufferConvertor ${PROTOBUF_LIBRARIES})
                                            
