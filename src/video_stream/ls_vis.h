
#include <vector>
#include <opencv2/opencv.hpp>
#include "structures/structs.h"

using namespace std;

namespace lsUtils{
    cv::Mat vis(cv::Mat img, int frame_id, vector<int> track_id, vector<vector<float>> head_boxes,
                vector<vector<float>> face_angle){
        for (int i = 0; i < head_boxes.size(); i++) {
            cv::Point p1, p2, p3, p4, p5;
            p1.x = head_boxes[i][0];
            p1.y = head_boxes[i][1];
            p2.x = head_boxes[i][2];
            p2.y = head_boxes[i][3];
            p3.x = p1.x + 5;
            p3.y = p1.y + 26;
            p4.x = p1.x + 5;
            p4.y = p1.y + 52;
            p5.x = p1.x + 5;
            p5.y = p1.y + 78;
            cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), 3, 4, 0);
            cv::putText(img, "Y: " + to_string(face_angle[i][0]), p3, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
            cv::putText(img, "P: " + to_string(face_angle[i][1]), p4, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
            cv::putText(img, "R: " + to_string(face_angle[i][2]), p5, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
        }
        return img;
    }

    cv::Mat vis_Box(cv::Mat img, vector<Box> hand_boxes){
        for (auto & box : hand_boxes){
            cv::Point p1, p2;
            p1.x = box.x1;
            p1.y = box.y1;
            p2.x = box.x2;
            p2.y = box.y2;
            cv::rectangle(img, p1, p2, cv::Scalar(0, 255, 255), 2, 1, 0);
        }
        cv::rectangle(img, cv::Point(150, 260), cv::Point(280, 359), cv::Scalar(0, 255, 0), 2, 1, 0);
        return img;
    }

    cv::Mat vis_hand_det_box(cv::Mat img){
        cv::rectangle(img, cv::Point(150, 260), cv::Point(280, 359), cv::Scalar(0, 255, 0), 2, 1, 0);
        return img;
    }
}
