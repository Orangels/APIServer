#pragma once
#include <vector>
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Common.h"

class Engine_Api
{
public:
    Engine_Api();
    ~Engine_Api();
    void detect_headface(cv::Mat &image, std::vector<int>& hf_boxs);
    void detect_headface(cv::Mat &image, void * hf_boxs);
    void get_angles(cv::Mat &image, std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& angles);
    void get_angles(cv::Mat &image, std::vector<std::vector<int>>& rects, std::vector<float>& angles);
    void get_angles(cv::Mat &image, std::vector<std::vector<int>>& rects, void * angles);
    void get_ageGender(cv::Mat &image, std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& infos);
    void get_ageGender(cv::Mat &image, std::vector<std::vector<int>>& rects, std::vector<float>& infos);
    void get_ageGender(cv::Mat &image, std::vector<std::vector<int>>& rects, void * infos);
private:
    CConfiger* m_pconfiger;
};