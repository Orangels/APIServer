#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "utils/split.h"

void split(cv::Mat &ori, std::vector<cv::Mat> & list_img, int cover_w){
    auto sz = ori.size();
    int w = sz.width;
    int h = sz.height;
    int o = cover_w;
    int dst_w = 2 * w / 3 + o * 2;
    int dst_h = h / 2;

    cv::Mat tmp_dst;
    cv::Mat tmp_img;

    cv::Mat dst_1;
    dst_1.create(dst_h, dst_w, ori.type());
    tmp_dst = dst_1(cv::Rect(o, 0, 2 * w / 3 + o , dst_h));
    tmp_img = ori(cv::Rect(0, 0, 2 * w / 3 + o, dst_h));
    tmp_img.copyTo(tmp_dst);
    tmp_dst = dst_1(cv::Rect(0, 0, o , dst_h));
    tmp_img = ori(cv::Rect(w - o, dst_h, o, dst_h));
    tmp_img.copyTo(tmp_dst);
    list_img.push_back(dst_1);
//    cv::imwrite("./img0.png",dst_1);

    cv::Mat dst_2;
    dst_2.create(dst_h, dst_w, ori.type());
    tmp_dst = dst_2(cv::Rect(0, 0, w / 3 + o , dst_h));
    tmp_img = ori(cv::Rect(2 * w / 3 - o, 0, w / 3 + o, dst_h));
    tmp_img.copyTo(tmp_dst);
    tmp_dst = dst_2(cv::Rect(w / 3 + o, 0, w / 3 +  o, dst_h));
    tmp_img = ori(cv::Rect(0, dst_h, w / 3 +  o, dst_h));
    tmp_img.copyTo(tmp_dst);
    list_img.push_back(dst_2);
//    cv::imwrite("./img2.png",dst_2);

    cv::Mat dst_3;
    dst_3.create(dst_h, dst_w, ori.type());
    tmp_dst = dst_3(cv::Rect(0, 0, 2 * w / 3 + o, dst_h));
    tmp_img = ori(cv::Rect(w / 3 - o, dst_h, 2 * w / 3 + o, dst_h));
    tmp_img.copyTo(tmp_dst);
    tmp_dst = dst_3(cv::Rect(2 * w / 3 + o, 0, o, dst_h));
    tmp_img = ori(cv::Rect(0, 0, o, dst_h));
    tmp_img.copyTo(tmp_dst);
    list_img.push_back(dst_3);
//    cv::imwrite("./img3.png",dst_3);
}