//
// Created by Orangels on 2020-04-24.
//

#ifndef FISHEYE_CAMERAHANDLER_H
#define FISHEYE_CAMERAHANDLER_H

#include "IMV1.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_gl_interop.h>


class cameraHandler : public IMV_CameraInterface {
public:
    cameraHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
                     int vOutput_width, int vOutput_height, unsigned char *vOutput_data,
                     float vCenter_zoom_angle);

    cameraHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
                  int vOutput_width, int vOutput_height, unsigned char *vOutput_data,
                  float vPerimeter_top_angle, float vPerimeter_bottom_angle);

    ~cameraHandler();

    int run(unsigned char *vInput_data);

    IMV_Buffer input_buffer;
    IMV_Buffer output_buffer;

private:
    cameraHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
                  int vOutput_width, int vOutput_height, unsigned char *vOutput_data);

};


class cameraGHandler : public IMV_CameraFlatSurfaceInterface{
public:
    cameraGHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
                  int vOutput_width, int vOutput_height, unsigned char *vOutput_data,
                  float vCenter_zoom_angle);

    cameraGHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
                  int vOutput_width, int vOutput_height, unsigned char *vOutput_data,
                  float vPerimeter_top_angle, float vPerimeter_bottom_angle);

    ~cameraGHandler();

    int run(unsigned char *vInput_data);

    void Process();

    IMV_Buffer input_buffer;
    IMV_Buffer output_buffer;

    // gpu address
    void *ipt_pixels_dyn_ls;
    // 0 :GL infer,  1: cpu infer
    int inferMode;
private:
    cameraGHandler(int vInput_width, int vInput_height, unsigned char *vInput_data,
                  int vOutput_width, int vOutput_height, unsigned char *vOutput_data);

    void InitGL();
    void UnpackToTexture();
    void RenderScene();
    void PackFromFramebuffer();

    int num_vertices;
    Vertex2D *vertices;
    Vertex2D *txcoords;

    GLuint tex_id;
    GLuint fbo_id;
    GLuint rbo_id;
    GLuint pbo_id[2];

    //paramers about cuda
    cudaGraphicsResource *resource;

    size_t size;

};


#endif //FISHEYE_CAMERAHANDLER_H
