#pragma once
#include <vector>

//#include "rtmpHandler.h"
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Common.h"

class SSD_Detection
{
public:
    SSD_Detection();
    ~SSD_Detection();
    void detect_hf(cv::Mat image, std::vector<int>& hf_boxs);
//    void detect_hf();
    void get_angles(cv::Mat image, std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& angles);
    void get_ageGender(cv::Mat image, std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& infos);
    void get_features(std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& features);
    //void detect_hand(cv::Mat &image, std::vector<float>& hand_boxs);
    //void detect_hop(cv::Mat &image, std::vector<float>& hop_boxs);
private:
    //CModelEngine *hf_m_pdetector, *hand_m_pdetector, *hop_m_pdetector, *fa_m_pdetector, *fr_m_pdetector;
    CConfiger* m_pconfiger;
    //CImage mhf_image;

    //unsigned char* mhf_gpuImage = NULL;

};