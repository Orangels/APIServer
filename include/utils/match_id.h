//
// Created by xxs on 2020/5/6.
//

#ifndef DHP_MATCH_ID_H
#define DHP_MATCH_ID_H

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "box_tracking.h"
#include "utils/track.h"
#include "structures/structs.h"
#include "structures/trajectory.h"

class Match_ID {
public:
    Match_ID(int head_track_mistimes, int w, int h, int dis);
    ~Match_ID();
    void MergeResult(cv::Mat img,cv::Mat &result_img, int num, vector<int> ret1,vector<int>ret2, vector<int> ret3,int without_id,Person_ID * Person_id);
    int find_pid(int );
    Track* head_tracker;
    vector<Tracks> all_tracks;
    Trajectory* trajectory;
    vector<int> match_tid;
    vector<int> match_pid;

};
#endif //DHP_MATCH_ID_H