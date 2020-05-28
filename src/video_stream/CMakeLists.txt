cmake_minimum_required(VERSION 3.5.1)
project(videoStream)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -g") #-O0 -ggdb

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if (${CMAKE_SYSTEM} MATCHES "Linux-4.9.140-tegra")
    message("On TEGRA PLATFORM.")
    add_definitions(-DPLATFORM_TEGRA)
    set(SYS_USR_LIB /usr/lib/aarch64-linux-gnu)
    set(SYS_LIB /lib/aarch64-linux-gnu)
    set(DS_LIB /opt/nvidia/deepstream/deepstream-4.0/lib)
    link_libraries(
            /opt/nvidia/deepstream/deepstream-4.0/lib/libnvdsgst_meta.so
            /opt/nvidia/deepstream/deepstream-4.0/lib/libnvds_meta.so
            /opt/nvidia/deepstream/deepstream-4.0/lib/libnvbufsurftransform.so
            /opt/nvidia/deepstream/deepstream-4.0/lib/libnvbufsurface.so
    )
else ()
    message("On X86 PLATFORM.")
    set(SYS_USR_LIB /usr/lib/x86_64-linux-gnu)
    set(SYS_LIB /lib/x86_64-linux-gnu)
endif ()

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        /usr/local/include
        ${CUDA_INCLUDE_DIRS}
        DSIncludes
        /usr/include/gstreamer-1.0
        /usr/include/glib-2.0
        ${SYS_USR_LIB}/glib-2.0/include
        /usr/local/cuda/targets/aarch64-linux/include/
#        /usr/local/ffmpeg/include
        /usr/local/include
)

file(GLOB sources *.cpp base64/*cpp)

add_library(videoStream SHARED ${sources})
target_link_libraries(videoStream ${OpenCV_LIBS}
        cuda
        ${CUDA_LIBRARIES}
        ${SYS_USR_LIB}/libgtk3-nocsd.so.0
        ${SYS_USR_LIB}/libgstreamer-1.0.so.0
        ${SYS_USR_LIB}/libgobject-2.0.so.0
        ${SYS_USR_LIB}/libglib-2.0.so.0
        ${SYS_LIB}/libc.so.6
        #/lib/ld-linux-aarch64.so.1
        ${SYS_LIB}/libdl.so.2
        ${SYS_LIB}/libpthread.so.0
        ${SYS_USR_LIB}/libgmodule-2.0.so.0
        ${SYS_LIB}/libm.so.6
        ${SYS_USR_LIB}/libffi.so.6
        ${SYS_LIB}/libpcre.so.3
        ${OpenCV_LIBS}
        cudart
#        /usr/local/ffmpeg/lib/libavcodec.so
#        /usr/local/ffmpeg/lib/libavdevice.so
#        /usr/local/ffmpeg/lib/libavfilter.so
#        /usr/local/ffmpeg/lib/libavformat.so
#        /usr/local/ffmpeg/lib/libavutil.so
#        /usr/local/ffmpeg/lib/libpostproc.so
#        #        /usr/local/ffmpeg/lib/libswresample.so
#        /usr/local/ffmpeg/lib/libswscale.so
        )


