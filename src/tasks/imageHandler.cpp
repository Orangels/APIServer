#include "tasks/imageHandler.h"
#include "singleton.h"
#include "config_yaml.h"


int64_t getCurrentTime_infer(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

imageHandler::imageHandler(){

}

imageHandler::imageHandler(int camId){
    yamlConfig *config_A = Singleton<yamlConfig>::GetInstance("/srv/media_info.yaml");
    auto       conf      = config_A->getConfig();

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

/**
 *
 * @param rc1 face
 * @param rc2 head
 * @param AJoin
 * @param AUnion
 * @return
 */
float computRectJoinUnion(const cv::Rect &rc1, const cv::Rect &rc2, float &AJoin, float &AUnion){
    cv::Point p1, p2;                 //p1为相交位置的左上角坐标，p2为相交位置的右下角坐标
    p1.x = std::max(rc1.x, rc2.x);
    p1.y = std::max(rc1.y, rc2.y);

    p2.x = std::min(rc1.x + rc1.width, rc2.x + rc2.width);
    p2.y = std::min(rc1.y + rc1.height, rc2.y + rc2.height);

    AJoin = 0;
    if (p2.x > p1.x && p2.y > p1.y)            //判断是否相交
    {
        AJoin = (p2.x - p1.x) * (p2.y - p1.y);    //求出相交面积
    }
    float A1 = rc1.width * rc1.height;
    float A2 = rc2.width * rc2.height;
    AUnion = (A1 + A2 - AJoin);                 //两者组合的面积

    if (AUnion > 0)
        //        return (AJoin / AUnion);                  //相交面积与组合面积的比例
        return (AJoin / A1);                  //相交面积与rc1面积的比例
    else
        return 0;
}

std::vector <std::vector<int>> imageHandler::bindFaceTracker(std::vector<int> vHf_boxs,
                                                             std::vector<int> tracking_result){

    std::vector <std::vector<int>> result_ldmk_boxes_tmp;
    std::vector <std::vector<int>> head_boxs_tmp;
    std::vector <std::vector<int>> ldmk_boxes_tmp;
    int                            head_num = 0;

    for (int i = 0; i < vHf_boxs.size(); i += 6) {
        if (hf_boxs[i + 5] == 1) {
            std::vector<int> box_tmp = {hf_boxs[i], hf_boxs[i + 1], hf_boxs[i + 2], hf_boxs[i + 3],
                                        tracking_result[head_num]};
            head_boxs_tmp.emplace_back(box_tmp);
            head_num += 1;
        } else if (hf_boxs[i + 5] == 2) {
            std::vector<int> box_tmp = {hf_boxs[i], hf_boxs[i + 1], hf_boxs[i + 2], hf_boxs[i + 3]};
            ldmk_boxes_tmp.emplace_back(box_tmp);
        }
    }

    //    bind face
    for (int i = 0; i < ldmk_boxes_tmp.size(); ++i) {
        int      face_xmin = ldmk_boxes_tmp[i][0];
        int      face_ymin = ldmk_boxes_tmp[i][1];
        int      face_xmax = ldmk_boxes_tmp[i][2];
        int      face_ymax = ldmk_boxes_tmp[i][3];
        cv::Rect face_rect = cv::Rect(face_xmin, face_ymin, face_xmax - face_xmin, face_ymax - face_ymin);

        for (int j = 0; j < head_boxs_tmp.size(); ++j) {
            int head_xmin = head_boxs_tmp[j][0];
            int head_ymin = head_boxs_tmp[j][1];
            int head_xmax = head_boxs_tmp[j][2];
            int head_ymax = head_boxs_tmp[j][3];
            int trackID   = head_boxs_tmp[j][4];

            cv::Rect head_rect  = cv::Rect(head_xmin, head_ymin, head_xmax - head_xmin, head_ymax - head_ymin);
            float    AJoin, AUnion;
            float    face_scale = computRectJoinUnion(face_rect, head_rect, AJoin, AUnion);
            if (face_scale >= 0.7) {
                if (face_tracker_count.find(trackID) == face_tracker_count.end()) { //不存在 key
                    if (result_ldmk_boxes_tmp.size()<8){
                        result_ldmk_boxes_tmp.emplace_back(ldmk_boxes_tmp[i]);
                        face_tracker_count[trackID] = frameCount;
                        cout << "new track id " << trackID <<" face ldmk" << endl;
                    }
                } else {
                    //                    fps: 10 , time: 1s
                    if (frameCount - face_tracker_count[trackID] > 10 * 1) {
                        if (result_ldmk_boxes_tmp.size()<8){
                            result_ldmk_boxes_tmp.emplace_back(ldmk_boxes_tmp[i]);
                            face_tracker_count[trackID] = frameCount;
                            cout << "update track id "<< trackID <<" face ldmk" << endl;
                        }
                    }
                }
                break;
            }
        }
    }

    return result_ldmk_boxes_tmp;

}

void imageHandler::run(cv::Mat &ret_img, int vFrameCount){
    frameCount = vFrameCount;
    int64_t detect_start = getCurrentTime_infer();

    hf_boxs.clear();
    ldmk_boxes.clear();
    rects.clear();
    angles.clear();

    trEngine->detect_headface(ret_img, hf_boxs);
    //    for (int i = 0; i < hf_boxs.size(); i += 6) {
    //        if (hf_boxs[i + 5] == 2) {
    //            std::vector<int> box_tmp = {hf_boxs[i], hf_boxs[i + 1], hf_boxs[i + 2], hf_boxs[i + 3]};
    //            if (ldmk_boxes.size() < 8) ldmk_boxes.emplace_back(box_tmp);
    //        }
    //    }

    headTracker->run(hf_boxs, 1);
    ldmk_boxes = bindFaceTracker(hf_boxs, headTracker->tracking_result);


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

    std::cout << "***********************" << endl;
    std::cout << "ldmk_boxes size -- " << ldmk_boxes.size() << endl;
    std::cout << "detection time cost -- " << detect_end - detect_start << endl;
    std::cout << "angle and age time cost -- " << ageGender_end - detect_end << endl;
    std::cout << "business time cost -- " << business_end - ageGender_end << endl;
    std::cout << "total time cost -- " << business_end - detect_start << endl;
    std::cout << "***********************" << endl;

}

void imageHandler::vis(cv::Mat &ret_img){
    if (angles.size() > 0) {
        vis_box_angles(ret_img, hf_boxs, angles, headTracker->tracking_result);
    } else {
        vis_box(ret_img, hf_boxs, headTracker->tracking_result);
    }

}