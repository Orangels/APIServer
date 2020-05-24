#ifndef BOX_TRACKING_H
#define BOX_TRACKING_H
#include "Hungarian/Hungarian.h"
#include <cmath>
using namespace  std;

namespace boxTracking{
    typedef struct Rect {
        int x;
        int y;
        float width;
        int height;
    } Rect;

    typedef struct matches {
        int det_id;
        int tracking_id;
        float iou;
        int obj_id;
    } matches;

    typedef struct tracker_info {
        int obj_id;
        bool iou_less;
        Rect rect;
        int mismatch_times;
        bool match_now;
    } tracker_info;


    class BoxTracker {
    public:
        BoxTracker(float iou_cost_weight_,float cost_th_,int max_mismatch_times_) {
                iou_cost_weight =iou_cost_weight_;
                cost_th =cost_th_ ;
                max_mismatch_times =max_mismatch_times_;
                out_boundary_delete = true;
                near_boundary_th =0.05;
                max_distance =3.0;
                iou_cost_weight =iou_cost_weight_;
                distance_cost_weight =0.5*max_distance;
                size_cost_weight=1;
                ratio_cost_weight=1;
                mismatchTimes_cost_weight=0.2;
                boundary_cost_weight=0.0;
                max_cost = iou_cost_weight + distance_cost_weight +size_cost_weight+ratio_cost_weight+mismatchTimes_cost_weight+boundary_cost_weight;

        }
        ~BoxTracker() {
        }
        vector<int> tracking_Frame_Hungarian(vector<Rect> detection_rects,int img_w_,int img_h_);




    private:
        vector<vector<double> > cal_costMatrix(vector<Rect> detection_rects);
        vector<matches> match_tracking_detinfo(vector<Rect> detection_rects);
        float intersection_over_union(Rect box1,Rect box2);
        float cal_total_cost(Rect box1,tracker_info info);
        bool near_boundary(Rect box1);
        bool out_boundary_delete;
        int frames_num =0;
        int current_max_id =0;
        int img_w;
        int img_h;
        float cost_th;
        float max_cost;
        vector<tracker_info> tracking_infos;
        int max_mismatch_times;
        float near_boundary_th;
        float iou_cost_weight;
        float distance_cost_weight;
        float size_cost_weight;
        float ratio_cost_weight;
        float mismatchTimes_cost_weight;
        float boundary_cost_weight;
        float max_distance;
    };


}

#endif // ALIGNMENT_H
