#ifndef IMAGEHANDLER_H
#define IMAGEHANDLER_H

#include <iostream>
#include <opencv2/core.hpp>
#include <vector>
#include "utils/track.h"
#include "utils/vis.h"
//#include "config.h"
#include "engineApi.h"
#include "EnginePy.hpp"
#include <unordered_map>

using namespace cv;

class imageHandler {
public:
    imageHandler();

    imageHandler(int camId);

    ~imageHandler();

    void run(cv::Mat &ret_img, int vFrameCount);

    void vis(cv::Mat &ret_img);

    void updateLosNum(int num);

    cv::Mat frame;
private:
    Engine_Api *trEngine;
    Track      *headTracker;
    Engine_api *pyEngineAPI;
    std::vector<int>                hf_boxs;
    std::vector<float>              vWangles;
    std::vector<float>              vWrects;
    std::vector<std::vector<int>>   ldmk_boxes;
    std::vector<std::vector<float>> angles;
    std::vector<std::vector<float>> rects;

    std::unordered_map<int, int> face_tracker_count;

    float *mWrects;
    float *mWangles;
    int frameCount;


    std::vector<std::vector<int>> bindFaceTracker(std::vector<int> vHf_boxs,
                                                  std::vector<int> tracking_result);
};


#endif