#ifndef IMAGEHANDLER_H
#define IMAGEHANDLER_H

#include <iostream>
#include <opencv2/core.hpp>
#include <vector>
#include "utils/track.h"
#include "utils/vis.h"
#include "config.h"
#include "engineApi.h"
#include "EnginePy.hpp"

using namespace cv;

class imageHandler{
    public:
        imageHandler();
        ~imageHandler();
        void run(cv::Mat ret_img);
        void vis(cv::Mat& ret_img);

        cv::Mat frame;

    private:
        Engine_Api *trEngine;
        Track *headTracker;
        Engine_api* pyEngineAPI;
        std::vector<int> hf_boxs;
        std::vector<std::vector<int>> ldmk_boxes;
        std::vector<std::vector<float>>angles;
        std::vector<std::vector<float>>rects;

        float* mWrects;
        float* mWangles;

};


#endif