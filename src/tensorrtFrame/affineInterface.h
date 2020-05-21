//
// Created by Li DC on 2019/6/25.
//
#pragma once
#ifndef AFFINEINTERFACE_H
#define AFFINEINTERFACE_H

#include <iostream>
#include<algorithm>
#include <string>
#include <cuda_runtime.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <time.h>

#define exitIfCudaError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
/**
 * An opaque value that represents a CUDA texture object
 */
typedef __device_builtin__ unsigned long long cudaTextureObject_t;

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s \n file: %s -line: %d\n", cudaGetErrorString(code), file, line);
        if (abort)exit(code);
    }
}


class Affine_ {
private:
    int MAX_NUM_BOXES, coeff_size, out_w, out_h;
    float *dev_coeff, *dev_5kpts;

    double *coeff_params;

    void get_5_points_general(float *kpts, std::vector<float> &points5, int num_kpts, int pointsNum, int);

    void get_transform(std::vector<double> &out, float *src_points, int num_boxes, int w = 112, int h = 112);

    void inv_transform(double *factor, float inv_coeff[][4], int num_pts);

    void re_malloc_coeff(int N);

public:
    cudaStream_t stream;
    cudaEvent_t tic, toc;
    bool show_eclipse_time;

    Affine_(int w = 112, int h = 112);

    ~Affine_();

    void affineInterface(int vBatchSize, float *vp68pointsOnGPU,
                         cudaTextureObject_t vSrcImg, int vWidthSrcImg, int vHeightSrcImg,
                         float *vopDst);


    void getGpuImage(int num_boxs, float *dev_img_out, float inv_coeff[][4]);//***, float *vpImageGPU = NULL
    char *get_random_name();

    void static_time(long kpts5, long coeff, long invs, long kernel, long total, int box);//
};

#endif