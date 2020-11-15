#include "tasks/imageHandler.h"
#include "singleton.h"
#include "config_yaml.h"

int64_t getCurrentTime_infer()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

imageHandler::imageHandler(){

}

imageHandler::imageHandler(int camId){
    yamlConfig *config_A = Singleton<yamlConfig>::GetInstance("/srv/media_info.yaml");
    auto conf = config_A->getConfig();

    trEngine    = new Engine_Api();
    headTracker = new Track(conf["CAM"][camId]["ALGORITHM"]["TRACKER"]["LOS_NUMBER"].as<int>(), 1280, 720);
    pyEngineAPI = new Engine_api("engine_api", camId);
}

imageHandler::~imageHandler(){
//    delete trEngine;
    delete headTracker;
    delete pyEngineAPI;
    std::cout << "del imageHandler" << endl;
}

void imageHandler::updateLosNum(int num){
    headTracker->tracker->max_mismatch_times = num;
}

void imageHandler::run(cv::Mat& ret_img){
    hf_boxs.clear();
    ldmk_boxes.clear();
    rects.clear();
    angles.clear();

    int64_t detect_start = getCurrentTime_infer();

    trEngine->detect_headface(ret_img, hf_boxs);
    for (int i = 0; i < hf_boxs.size(); i += 6) {
        if (hf_boxs[i + 5] == 2) {
            std::vector<int> box_tmp = {hf_boxs[i], hf_boxs[i + 1], hf_boxs[i + 2], hf_boxs[i + 3]};
            if (ldmk_boxes.size() < 8) ldmk_boxes.emplace_back(box_tmp);
        }
    }

    headTracker->run(hf_boxs, 1);

    int64_t detect_end = getCurrentTime_infer();

    if (ldmk_boxes.size() > 0) {
        //        trEngine->get_angles(ret_img,ldmk_boxes,angles);
        //        trEngine->get_ageGender(ret_img,ldmk_boxes,rects);
        trEngine->get_angles(ret_img, ldmk_boxes, vWangles);
        trEngine->get_ageGender(ret_img, ldmk_boxes, vWrects);
    }

    int64_t ageGender_end = getCurrentTime_infer();

    pyEngineAPI->get_result(ret_img, hf_boxs, headTracker->tracking_result, headTracker->delete_tracking_id,
                            ldmk_boxes, vWangles, vWrects);

    int64_t business_end = getCurrentTime_infer();

//    std::cout << "***********************" << endl;
//    std::cout << "ldmk_boxes size -- " << ldmk_boxes.size() << endl;
//    std::cout << "detection time cost -- " << detect_end - detect_start << endl;
//    std::cout << "angle and age time cost -- " << ageGender_end - detect_end << endl;
//    std::cout << "business time cost -- " << business_end - ageGender_end << endl;
//    std::cout << "total time cost -- " << business_end - detect_start << endl;
//    std::cout << "***********************" << endl;

}

void imageHandler::vis(cv::Mat &ret_img){
    if (angles.size() > 0) {
        vis_box_angles(ret_img, hf_boxs, angles, headTracker->tracking_result);
    } else {
        vis_box(ret_img, hf_boxs, headTracker->tracking_result);
    }

}