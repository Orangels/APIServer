#pragma once

#include <iostream>
#include <stdio.h>
#include <unordered_map>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "cudaUtility.h"
#include "tensorNet.h"
#include "Common.h"
#include "FileFunction.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/imgproc/types_c.h>

struct LayerInfo
{
    std::vector<int> dim;
    std::string name;
    int index;
    int size;
};

std::vector<int> get_dim_size(Dims dim);

int total_size(std::vector<int> dim);

template<typename T>
void loadImage(int vHeight, int vWidth, T* vsrc, float* dst, int vBatchSize);

void _outputSSD(float xMin, float yMin, float xMax, float yMax, float vConfidence, int vClassID);

void setMaxCameraNum(int vMaxThreadNum);

void initialize(const char* vpModelPrefixFileName, char* vpFunctionName);

int _inference(int vBatchSize, int* vpImgData, void* vopPoint, const std::string& vFunctionName);

int _inference(int vBatchSize, float* vpImgData, void* vopPoint, const std::string& vFunctionName);

int keyPointInference(int vBatchSize, int* vpImgData, void* vopPoint);

int ageGenderInference(int vBatchSize, int* vpImgData, void* vopPoint);

int verifyFaceInference(int vBatchSize, float* vpImgData, void* vopPoint);

int verifyFaceInference2(int vBatchSize, float* vpImgData, void* vopPoint);

int ssdInference(int vBatchSize, int vHeight, int vWidth, uchar* vpImgData,
                 void* vopBoxScoreCls, int* vopNumDetections);


void trans_date(cv::Mat & faceImg, std::vector<uchar> &imgs, int batch_size);
