#ifndef IMAGEHANDLER_H
#define IMAGEHANDLER_H

#include <iostream>
#include <opencv2/core.hpp>
#include <vector>
#include "utils/track.h"
#include "utils/vis.h"
#include "config.h"
#include "detection.h"

using namespace cv;

class imageHandler{
    public:
        imageHandler();
        ~imageHandler();
        void run(cv::Mat ret_img);
        void vis(cv::Mat& ret_img);

        cv::Mat frame;

    private:
        SSD_Detection *trEngine;
        Track *headTracker;
        std::vector<int> hf_boxs;
        std::vector<std::vector<int>> ldmk_boxes;
        std::vector<std::vector<float>>rects;
        std::vector<std::vector<float>>angles;

};


#endif