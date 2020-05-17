#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils/vis.h"

void vis_box(cv::Mat &img, std::vector<int> boxes){
    // draw img
    for (int i = 0;i <boxes.size(); i+=6){
        // vis_th
        //cout<<boxes[i+2]<<" "<<boxes[i+3]<<" " <<boxes[i+4]<<" " <<boxes[i+5]<<endl;
        std::string text = std::to_string(boxes[i+1]);
        std::vector<cv::Scalar> colors = {cv::Scalar(0,0,0),cv::Scalar(0,255,0),cv::Scalar(255,0,0),cv::Scalar(0,0,255)};
        cv::rectangle(img,cv::Point(boxes[i+2],boxes[i+3]), cv::Point(boxes[i+4], boxes[i+5]),colors[boxes[i]],1,1,0);
        cv::putText(img, text, cv::Point(boxes[i+2],boxes[i+3]), cv::FONT_HERSHEY_COMPLEX, 0.5, colors[boxes[i]]);
    }

}