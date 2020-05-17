#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

void split(cv::Mat &ori, std::vector<cv::Mat> &list_img, int cover_w);