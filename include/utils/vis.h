#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

void vis_box(cv::Mat & img, std::vector<int> box);
void vis_box(cv::Mat & img, std::vector<int> box, std::vector<int> trackID);