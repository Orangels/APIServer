#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>

void vis_box(cv::Mat & img, std::vector<int> box);
void vis_box(cv::Mat & img, std::vector<int> box, std::vector<int> trackID);
void vis_box_angles(cv::Mat & img, std::vector<int> box, std::vector<std::vector<float>>angles, std::vector<int> trackID);