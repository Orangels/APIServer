//
// Created by xinxuehsi  on 2020/5/1.
//

#include "utils/track.h"

Track::Track(int head_track_mistimes, int w, int h) {
    tracker = new BoxTracker(4.0, 1.0, head_track_mistimes);
    img_w = w;
    img_h = h;
}

Track::~Track() = default;

void Track::run(vector<Box> boxes_xxyy) {
    int num_del;
    Rect rect;
    vector<int> result;
    detection_rects.clear();
    tracking_result.clear();
    delete_tracking_id.clear();
    for (auto &box_xxyy : boxes_xxyy){
        rect.x = box_xxyy.x1;
        rect.y = box_xxyy.y1;
        rect.width = box_xxyy.x2 - box_xxyy.x1;
        rect.height = box_xxyy.y2 - box_xxyy.y1;
        detection_rects.push_back(rect);
    }
    result = tracker->tracking_Frame_Hungarian(detection_rects, img_w, img_h);
    for(int i = 0; i< boxes_xxyy.size(); i++){
        tracking_result.push_back(result[i]);
    }
    if (boxes_xxyy.size() < result.size()){
        num_del = result[boxes_xxyy.size()];
        for(int i = 0; i< num_del; i++){
            delete_tracking_id.push_back(result[i + boxes_xxyy.size() + 1]);
        }
    }
}