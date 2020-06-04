#include "tasks/imageHandler.h"

imageHandler::imageHandler(){

}

imageHandler::imageHandler(int camId){
    Cconfig labels = Cconfig("../cfg/process.ini");

    trEngine = new Engine_Api();
    headTracker = new Track(stoi(labels["HEAD_TRACK_MISTIMES"]), stoi(labels["OUT_W"]), stoi(labels["OUT_H"]));
    pyEngineAPI = new Engine_api("engine_api", camId);
}

imageHandler::~imageHandler(){
    delete trEngine;
    delete headTracker;
    delete pyEngineAPI;
    std::cout << "del imageHandler" << endl;
}

void imageHandler::run(cv::Mat ret_img){
    hf_boxs.clear();
    ldmk_boxes.clear();
    rects.clear();
    angles.clear();
    
    trEngine->detect_headface(ret_img, hf_boxs);
    for (int i = 0; i < hf_boxs.size(); i+=6) {
        if (hf_boxs[i+5]==2){
            std::vector<int> box_tmp = {hf_boxs[i],hf_boxs[i+1],hf_boxs[i+2],hf_boxs[i+3]};
            if (ldmk_boxes.size() < 8) ldmk_boxes.emplace_back(box_tmp);
        }
    }

    headTracker->run(hf_boxs, 1);

    if (ldmk_boxes.size()>0){
//        trEngine->get_angles(ret_img,ldmk_boxes,angles);
//        trEngine->get_ageGender(ret_img,ldmk_boxes,rects);
        trEngine->get_angles(ret_img,ldmk_boxes,vWangles);
        trEngine->get_ageGender(ret_img,ldmk_boxes,vWrects);
    }

    vector<int> vret0;
    pyEngineAPI->get_result(ret_img.clone(), hf_boxs, headTracker->tracking_result, headTracker->delete_tracking_id, ldmk_boxes, vWangles, vWrects);

}

void imageHandler::vis(cv::Mat& ret_img){
    if (angles.size()>0){
        vis_box_angles(ret_img, hf_boxs, angles, headTracker->tracking_result);
    } else{
        vis_box(ret_img, hf_boxs, headTracker->tracking_result);
    }

}