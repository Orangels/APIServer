#include "box_tracking.h"
//#include "face_common.h"

using namespace std;
using namespace boxTracking;

float BoxTracker::intersection_over_union(Rect box1,Rect box2)
{
    int x1 = std::max(box1.x,box2.x);
    int y1 = std::max(box1.y,box2.y);
    int x2 = std::min((box1.x + box1.width),(box2.x + box2.width));
    int y2 = std::min((box1.y + box1.height),(box2.y + box2.height));
    float over_area = std::max((x2 - x1),0) * std::max((y2 - y1),0);
    float iou = float(over_area)/(box1.width * box1.height + box2.width * box2.height-over_area);
    return iou;
}

bool BoxTracker::near_boundary(Rect box1)
{

    //cout<< img_w <<endl;
    int x2 = box1.x+box1.width;
    int y2 = box1.y+box1.height;
    bool x1_boundary = float(box1.x)/img_w<near_boundary_th;
    bool y1_boundary = float(box1.y)/img_h<near_boundary_th;
    bool x2_boundary = float(img_w - x2)/img_w<near_boundary_th;
    bool y2_boundary = float(img_h - y2)/img_h<near_boundary_th;

    return x1_boundary || y1_boundary || x2_boundary || y2_boundary;
}


float BoxTracker::cal_total_cost(Rect box1,tracker_info info)
{
    Rect box2 = info.rect;
    float total_cost;

    int s1 = box1.width * box1.height;
    int s2 = box2.width * box2.height;
    float size_cost = float(std::fabs(s1-s2))/(std::min(s1,s2));//(0,++)

    float xy_ratio1 = float(box1.width)/box1.height;
    float xy_ratio2 = float(box2.width)/box2.height;
    float ratio_cost = std::max(xy_ratio1,xy_ratio2)/std::min(xy_ratio1,xy_ratio2)-1;//[0,++)

    float center_x1 = box1.x+box1.width/2;
    float center_y1 = box1.y+box1.height/2;
    float center_x2 = box2.x+box2.width/2;
    float center_y2 = box2.y+box2.height/2;
    float distance_cost = sqrt((center_x1-center_x2)*(center_x1-center_x2) + (center_y1-center_y2)*(center_y1-center_y2));//[0,++)
    distance_cost = distance_cost / sqrt(std::min(s1,s2)) / max_distance;//[0,++)

    // time_cost
    float mismatchTimes_cost = float(info.mismatch_times /max_mismatch_times);//[0,1)

    // boundary_cost
    float boundary_cost = float(near_boundary(box2));


    float iou_cost = 1- intersection_over_union(box1,box2);


    //cout<<"size_cost= "<<size_cost<<" ratio_cost= "<<ratio_cost<<" distence_cost= "<<distence_cost<<" iou_cost= "<<iou_cost<<endl;

    if(size_cost>1 || ratio_cost>1 || distance_cost > 1){
        total_cost = max_cost;
        return total_cost;
    }

    total_cost = iou_cost*iou_cost_weight + size_cost*size_cost_weight + ratio_cost*ratio_cost_weight + \
            distance_cost*distance_cost_weight + mismatchTimes_cost*mismatchTimes_cost_weight + boundary_cost*boundary_cost_weight;

    return total_cost;
}


vector<vector<double> > BoxTracker::cal_costMatrix(vector<Rect> detection_rects){
    vector<vector<double> > total_cost_matrixs;
    for(int i =0;i<detection_rects.size();i++){
        vector<double> total_cost_matrix;
        for(int j =0;j<tracking_infos.size();j++){
            //float total_cost =1- intersection_over_union(detection_rects[i],tracking_infos[j].rect); //Iou COST
            float total_cost = cal_total_cost(detection_rects[i],tracking_infos[j]);
            total_cost_matrix.push_back(total_cost);
        }
        total_cost_matrixs.push_back(total_cost_matrix);
    }
    return total_cost_matrixs;
}


vector<matches> BoxTracker::match_tracking_detinfo(vector<Rect> detection_rects){
    vector<vector<double> > total_cost_matrixs =cal_costMatrix(detection_rects);
    HungarianAlgorithm HungAlgo;
    vector<int> assignment_tracking_id;
    if(detection_rects.size())
        double total_cost = HungAlgo.Solve(total_cost_matrixs, assignment_tracking_id);
    vector<matches> mathch_rects;

    for(int i =0;i<tracking_infos.size();i++)
    {
        tracking_infos[i].match_now =false;
    }

    //detection_rects.size() == assignment_tracking_id.size()
    for(int i =0;i<detection_rects.size();i++)
    {
        int tracking_id = assignment_tracking_id[i];
        matches mathch_rect;
        mathch_rect.det_id = i;
        if(tracking_id>=0){
            mathch_rect.tracking_id = tracking_id;
            mathch_rect.iou = 1-total_cost_matrixs[i][tracking_id];
            mathch_rect.obj_id = tracking_infos[tracking_id].obj_id;
            //if(mathch_rect.iou < IoU_th){
            if(total_cost_matrixs[i][tracking_id] >= max_cost* cost_th){
                mathch_rect.obj_id = -1;
            }
            else{
                tracking_infos[tracking_id].match_now =true;
                tracking_infos[tracking_id].mismatch_times =0;
            }
        }
        else{
            mathch_rect.obj_id = -1;
        }

        mathch_rects.push_back(mathch_rect);
    }

    return mathch_rects;

}



vector<int> BoxTracker::tracking_Frame_Hungarian(vector<Rect> detection_rects,int img_w_,int img_h_){

    img_w = img_w_;
    img_h = img_h_;
//     std::cout<< "_________________"<<std::endl;
    vector <int> tracking_result;
    vector <int> tracking_end_id;
    //tracking
    if (frames_num==0) {
        for (int j = 0; j < detection_rects.size(); j++){
            tracker_info track_info;
            track_info.rect = detection_rects[j];
            track_info.obj_id = j;
            track_info.mismatch_times =0;
            track_info.iou_less = false;
            tracking_infos.push_back(track_info);
            tracking_result.push_back(current_max_id);
            current_max_id +=1;
        }
    }
    else{
        vector<matches> match_rectinfos = match_tracking_detinfo(detection_rects);
        for(int j=0;j<match_rectinfos.size();j++){
            if(match_rectinfos[j].obj_id == -1){
                Rect rect = detection_rects[match_rectinfos[j].det_id];


                current_max_id += 1;
                match_rectinfos[j].obj_id = current_max_id;
                tracker_info track_info;
                track_info.rect = rect;
                track_info.obj_id = current_max_id;
                track_info.mismatch_times =0;
                track_info.iou_less = false;
                tracking_infos.push_back(track_info);
            }
            else{
                tracking_infos[match_rectinfos[j].tracking_id].rect = detection_rects[match_rectinfos[j].det_id];
                //rectangle( image_input, tracking_infos[match_rectinfos[j].tracking_id].rect, Scalar( 0, 255, 255 ), 6, 8 );
            }

        }


        int tracking_infos_size =tracking_infos.size();
        for(int id = 0;id < tracking_infos_size; id++){

            if(out_boundary_delete){
                //如果跟踪器出边缘，快时间内删除跟踪器
                if(near_boundary(tracking_infos[id].rect) && tracking_infos[id].mismatch_times<=(max_mismatch_times*0.9-1))
                    tracking_infos[id].mismatch_times = max_mismatch_times*0.9-1;
            }

            if(tracking_infos[id].match_now){
                tracking_infos[id].mismatch_times =0;
            }
            else{
                tracking_infos[id].mismatch_times ++;
            }

            // 如果多次跟踪不到删除多余的跟踪器
            //或多个跟踪器重叠较大，删除(合并)多余的跟踪器
            if(tracking_infos[id].iou_less == false){
                for(int id_ = id+1;id_ < tracking_infos_size; id_++){
                    if(intersection_over_union(tracking_infos[id].rect,tracking_infos[id_].rect)>0.1){
                        tracking_infos[id].iou_less = true;
                        tracking_infos[id_].iou_less = true;
                    }
                }
            }
            // bool is_less =(!tracking_infos[id].match_now && tracking_infos[id].iou_less) ;
            bool is_less =(tracking_infos[id].iou_less) ;
            bool is_time_out = tracking_infos[id].mismatch_times > max_mismatch_times;
            //if(is_less | is_time_out){
            if(is_time_out){
                //cout<< "erase Tracker: id = "<< tracking_infos[id].obj_id << endl;
                tracking_end_id.push_back(tracking_infos[id].obj_id);
                tracking_infos.erase(tracking_infos.begin() + id);
                id --;
                tracking_infos_size --;

            }
        }

        //cout<<"max_id= "<<current_max_id<<"  tracker_num= "<<tracking_infos.size()<<endl;
        for(int j=0;j<match_rectinfos.size();j++)
            tracking_result.push_back(match_rectinfos[j].obj_id);

        tracking_result.push_back(tracking_end_id.size());
        tracking_result.insert(tracking_result.end(),tracking_end_id.begin(),tracking_end_id.end());
    }

    frames_num ++;
    return tracking_result;

}


