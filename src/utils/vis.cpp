#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils/vis.h"

using namespace std;

void vis_box(cv::Mat &img, std::vector<int> boxes){
    // draw img
    for (int i = 0;i <boxes.size(); i+=6){
        // vis_th
        std::string text = std::to_string(boxes[i+4]);
        std::vector<cv::Scalar> colors = {cv::Scalar(0,0,0),cv::Scalar(0,255,0),cv::Scalar(255,0,0),cv::Scalar(0,0,255)};
        cv::rectangle(img,cv::Point(boxes[i],boxes[i+1]), cv::Point(boxes[i+2], boxes[i+3]),colors[boxes[5]],1,1,0);
        cv::putText(img, text, cv::Point(boxes[i],boxes[i+1]), cv::FONT_HERSHEY_COMPLEX, 0.5, colors[boxes[5]]);
    }

}


void vis_box(cv::Mat & img, std::vector<int> boxes, std::vector<int> trackID){
    int trackNum = 0;
    for (int i = 0;i <boxes.size(); i+=6){
        // vis_th
        if (boxes[i+5]==1){
            std::string text = std::to_string(trackID[trackNum]);
            std::vector<cv::Scalar> colors = {cv::Scalar(0,0,0),cv::Scalar(0,255,0),cv::Scalar(255,0,0),cv::Scalar(0,0,255)};
            cv::rectangle(img,cv::Point(boxes[i],boxes[i+1]), cv::Point(boxes[i+2], boxes[i+3]),colors[boxes[i+5]],3,1,0);
            cv::putText(img, text, cv::Point(boxes[i],boxes[i+1]), cv::FONT_HERSHEY_COMPLEX, 1, colors[boxes[i+5]], 2);
            trackNum ++;
        } else{
            std::vector<cv::Scalar> colors = {cv::Scalar(0,0,0),cv::Scalar(0,255,0),cv::Scalar(255,0,0),cv::Scalar(0,0,255)};
            cv::rectangle(img,cv::Point(boxes[i],boxes[i+1]), cv::Point(boxes[i+2], boxes[i+3]),colors[boxes[i+5]],3,1,0);

//            std::string text = std::to_string(trackID[i / 6]);
//            cv::putText(img, text, cv::Point(boxes[i],boxes[i+1]), cv::FONT_HERSHEY_COMPLEX, 1, colors[boxes[i+5]], 2);
        }
    }
}

void vis_box_angles(cv::Mat & img, std::vector<int> boxes, std::vector<std::vector<float>>angles, std::vector<int> trackID){
    int trackNum = 0;
    for (int i = 0;i <boxes.size(); i+=6){
        // vis_th
        if (boxes[i+5]==1){
            std::string text = std::to_string(trackID[trackNum]);
            std::vector<cv::Scalar> colors = {cv::Scalar(0,0,0),cv::Scalar(0,255,0),cv::Scalar(255,0,0),cv::Scalar(0,0,255)};
            cv::rectangle(img,cv::Point(boxes[i],boxes[i+1]), cv::Point(boxes[i+2], boxes[i+3]),colors[boxes[i+5]],3,1,0);
            cv::putText(img, text, cv::Point(boxes[i],boxes[i+1]), cv::FONT_HERSHEY_COMPLEX, 1, colors[boxes[i+5]], 2);
            trackNum ++;
        } else{
            cv::Point p1, p2, p3, p4, p5;
            p1.x = boxes[i];
            p1.y = boxes[i+1];
            p2.x = boxes[i+2];
            p2.y = boxes[i+3];
            p3.x = p1.x + 5;
            p3.y = p1.y + 26;
            p4.x = p1.x + 5;
            p4.y = p1.y + 52;
            p5.x = p1.x + 5;
            p5.y = p1.y + 78;
            std::vector<cv::Scalar> colors = {cv::Scalar(0,0,0),cv::Scalar(0,255,0),cv::Scalar(255,0,0),cv::Scalar(0,0,255)};
            cv::putText(img, "Y: " + to_string(angles[i][0]), p3, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
            cv::putText(img, "P: " + to_string(angles[i][1]), p4, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
            cv::putText(img, "R: " + to_string(angles[i][2]), p5, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
            cv::rectangle(img,cv::Point(boxes[i],boxes[i+1]), cv::Point(boxes[i+2], boxes[i+3]),colors[boxes[i+5]],3,1,0);

        }
    }

}