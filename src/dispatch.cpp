//
// Created by xinxueshi on 2020/4/28.
//  https://blog.csdn.net/yiyouxian/article/details/51993524
//
#include <iostream>
#include <iomanip>
//#include <numpy/arrayobject.h>
#include "dispatch.h"
#include "config.h"
#include "Common.h"
#include "utils/mat2numpy.h"
#include "utils/vis.h"
#include "utils/split.h"
#include "utils/track.h"
#include "utils/match_id.h"
#include <sys/time.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"



using namespace std;

static Person_ID * Person_id = new Person_ID();

int64_t getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}


Dispatch::Dispatch()
{
    cout << "start Init Dispatch" << endl;
    mQueueLen = stoi(labels["QLEN"]);
    frames_skip = stoi(labels["FRAMES_SKIP"]); //2
    rtmp_mode = stoi(labels["RTMP_MODE"]); //0
    int fps = stoi(labels["FPS"]);  //20
    int out_w = stoi(labels["OUT_W"]);  //2880
    int out_h = stoi(labels["OUT_H"]);  //960

    string path_0 = labels["video_path_0"];
    string path_1 = labels["video_path_1"];

    mCamLive.resize(4);
    mCamPath.resize(4);
    mQueueCam.resize(4);
    mQueue_rtmp.resize(4);

    mCon_not_full.resize(4);
    mCon_not_empty.resize(4);
    mCon_rtmp.resize(4);
    mConMutexCam.resize(4);
    mConMutexRTMP.resize(4);
    mRtmpMutex.resize(4);

    mCamLive = {true, true, false, false};

    mCon_not_full = { &vCon_not_full_0, &vCon_not_full_1, &vCon_not_full_2, &vCon_not_full_3 };
    mCon_not_empty = { &vCon_not_empty_0, &vCon_not_empty_1, &vCon_not_empty_2, &vCon_not_empty_3};
    mCon_rtmp = {&vCon_rtmp_0, &vCon_rtmp_1, &vCon_rtmp_2, &vCon_rtmp_3};
    mConMutexCam = {&vConMutexCam_0, &vConMutexCam_1, &vConMutexCam_2, &vConMutexCam_3};
    mConMutexRTMP = {&vConMutexRTMP_0, &vConMutexRTMP_1, &vConMutexRTMP_2, &vConMutexRTMP_3};
    mRtmpMutex = {&vRtmpMutex_0, &vRtmpMutex_1, &vRtmpMutex_2, &vRtmpMutex_3};


//    for (int i = 0; i < 4; ++i) {
//        cout << i << " : " << mCon_not_full[i] << endl;
//        cout << i << " : " << mCon_not_empty[i] << endl;
//        cout << i << " : " << mCon_rtmp[i] << endl;
//        cout << i << " : " << mConMutexCam[i] << endl;
//        cout << i << " : " << mConMutexRTMP[i] << endl;
//        cout << i << " : " << mRtmpMutex[i] << endl;
//    }

    mRtmpImg.resize(4);

    mCamPath[0] = path_0;
    mCamPath[1] = path_1;


    if (rtmp_mode){
        std::string rtmpPath_0 = labels["RTMP_PATH_0"]; //  rtmp://127.0.0.1:1935/hls/room
        std::string rtmpPath_1 = labels["RTMP_PATH_1"];
        cout << rtmpPath_0 << endl;
        cout << rtmpPath_1 << endl;
//        ls_handler_front = rtmpHandler("",rtmpPath_0,out_w,out_h,fps);
//        ls_handler_mid = rtmpHandler("",rtmpPath_1,out_w,out_h,fps);
    }


//    int tracker_mode = stoi(labels["TRACK_MODE"]);
//    if (stoi(labels["inference_switch"])){
//        int ret = 0;
//        Py_Initialize();
//        if (!Py_IsInitialized())
//        {
//            LOG_DEBUG("Py_Initialize error, return\n");
//        }
//
//        PyEval_InitThreads();
//        int nInit = PyEval_ThreadsInitialized();
//        if (nInit)
//        {
//            LOG_DEBUG("PyEval_SaveThread\n");
//            PyEval_SaveThread();
//        }
//        if (tracker_mode == 0){
//            pyEngineAPI_0 =  new Engine_api("engine_api");
//            pyEngineAPI_1 =  new Engine_api("engine_api");
//            pyEngineAPI_2 =  new Engine_api("engine_api");
//        } else{
//            cout << "init py start" << endl;
//            pyEngineAPI_0 =  new Engine_api("tracker_api");
//            pyEngineAPI_1 =  new Engine_api("tracker_api");
//            pyEngineAPI_2 =  new Engine_api("tracker_api");
//            cout << "init py end" << endl;
//        }
//
//        matcher = new Match_ID(stoi(labels["HEAD_TRACK_MISTIMES"]), stoi(labels["IMAGE_W"]), stoi(labels["IMAGE_H"]), stoi(labels["DIS"]));
//    }
    cout << "end Init Dispatch" << endl;

}
Dispatch::~Dispatch() = default;

void Dispatch::run()
{

}
void Dispatch::test()
{
    cout<< "test in "<<endl;
}

void Dispatch::RPCServer(){

}

void Dispatch::ConsumeRTMPImage(int mode){

    cv::Mat img;
    int num = 0;

    mutex *lock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait;
    mutex* rtmpLock;

    lock = mConMutexRTMP[mode];
    queue = &mQueue_rtmp[mode];
    con_v_wait = mCon_rtmp[mode];
    rtmpLock = mRtmpMutex[mode];

    cout << "ConsumeRTMPImage  start " << endl;
    cout << lock << endl;
    cout << con_v_wait << endl;
    cout << rtmpLock << endl;
    cout << "ConsumeRTMPImage  end " << endl;



    while (mCamLive[mode]) {
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->empty()) {
//            std::cout << "Consumer RTMP " << mode << " -- " << num <<" is waiting for items...\n";
            con_v_wait->wait(guard);
        }
        int64_t start_read = getCurrentTime();
        img = queue->front().clone();
        queue->pop();
        guard.unlock();
        rtmpLock->lock();
//        TODO 推流逻辑
//        rtmpHandler->pushRTMP(img);
        rtmpLock->unlock();
        num++;
    }
}

void Dispatch::ProduceImage(int mode){

    cv::VideoCapture cam;
    cv::Mat frame;

    string path = "";
    mutex *lock;
    mutex *rtmpLock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait, *con_v_notification;
//    rtmpHandler* rtmpHandler;
    cv::Mat *rtmp_img;


    path = mCamPath[mode];
    lock = mConMutexCam[mode];
//    lock = &myMutex_front;
    queue = &mQueueCam[mode];
    con_v_wait = mCon_not_full[mode];
    con_v_notification = mCon_not_empty[mode];
//            rtmpHandler = &ls_handler_front;
    rtmpLock = mRtmpMutex[mode];
    rtmp_img = &mRtmpImg[mode];

    cout << "ProduceImage  start " << endl;
    cout << lock << endl;
    cout << con_v_wait << endl;
    cout << con_v_notification << endl;
    cout << rtmpLock << endl;
    cout << "ProduceImage  end " << endl;

    cam.open(path);
    cout << "mode "<< mode << " open camera suc "<< getCurrentTime() << endl;

    if (!cam.isOpened())
    {
        cout << "cam open failed!" << endl;
        return;
    }


    if (!cam.isOpened())
    {
        cout << "cam open failed!" << endl;
        return;
    }

    cout << "mode "<< mode << "camera total suc "<< getCurrentTime() << endl;

    int sum = 0;
    int num = 0;
    int64_t end = getCurrentTime();

    int circle_i = 0;
    while (mCamLive[mode]) {
        circle_i ++;
        cam.read(frame);
//        cout << "ProduceImage "<< mode << " img "<< circle_i << " cost : " << getCurrentTime()-end << endl;
        end = getCurrentTime();

        if (circle_i % frames_skip != 0 ){
            if (rtmp_mode == 1){
                rtmpLock->lock();
//                rtmpHandler->pushRTMP(*rtmp_img);
                rtmpLock->unlock();
            }
            continue;
        }

        num ++;
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->size() >=  mQueueLen) {
//            std::cout << "Produce " << mode <<" is waiting for items...\n";
            con_v_wait->wait(guard);
        }


        if (stoi(labels["SAVE_IAMGE"]))
        {
            string img_path, img_path_center, img_path_ori;
            img_path_ori = "./imgs_ori/" + to_string(num+10000) + "_" + to_string(mode) + ".jpg";
            cv::imwrite(img_path_ori, frame);
        }

        queue->push(frame);
        con_v_notification->notify_all();
        guard.unlock();
    }

}


void Dispatch::ConsumeImage(int mode){
    int out_w = stoi(labels["OUT_W"]);  //2880
    int out_h = stoi(labels["OUT_H"]);  //960

    int num = 0;

    cv::Mat frame;

    mutex *lock;
    queue<cv::Mat> *rtmpQueue;
    queue<cv::Mat> *queue;
    condition_variable *con_rtmp;
    condition_variable *con_v_wait, *con_v_notification;
    cv::Mat* rtmp_img;
    cv::Mat ret_img;

    lock = mConMutexCam[mode];
//    lock = &myMutex_front;
    queue = &mQueueCam[mode];
    con_v_wait = mCon_not_empty[mode];
    con_v_notification = mCon_not_full[mode];
//            rtmpHandler = &ls_handler_front;
    con_rtmp = mCon_rtmp[mode];
    rtmp_img = &mRtmpImg[mode];

    cout << "ConsumeImage  start " << endl;
    cout << lock << endl;
    cout << con_v_wait << endl;
    cout << con_v_notification << endl;
    cout << con_rtmp << endl;
    cout << "ConsumeImage  end " << endl;


    while (mCamLive[mode]){
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->empty()) {
//            std::cout << "Consumer " << mode <<" is waiting for items...\n";
            con_v_wait->wait(guard);
        }

        frame = queue->front();
        queue->pop();
        con_v_notification->notify_all();
        guard.unlock();


        //        TODO 业务逻辑
        ret_img = frame.clone();

        if (stoi(labels["SAVE_IAMGE"]))
        {
            string img_path, img_path_center, img_path_ori;
            img_path = "./imgs/" + to_string(num+10000) + "_" + to_string(mode) + ".jpg";
            cv::imwrite(img_path, frame);
        }

        if (rtmp_mode == 1) {
            // 推流

            cv::Mat rtmp_frame;
            cv::resize(ret_img, rtmp_frame, cv::Size(out_w, out_h));

            cv::Mat frame_clone = rtmp_frame.clone();
            *rtmp_img = frame_clone;
            rtmpQueue->push(*rtmp_img);
            con_rtmp->notify_all();
        }
//        cout << "Consumer number : " << num << endl;
        num++;
    }

}

void Dispatch::multithreadTest(){

    vector<thread> threadArr;
    for (int i = 0; i < mCamLive.size(); ++i) {
        if (mCamLive[i]){

            threadArr.emplace_back(&Dispatch::ProduceImage, this, i);
            threadArr.emplace_back(&Dispatch::ConsumeImage, this, i);
            threadArr.emplace_back(&Dispatch::ConsumeRTMPImage, this, i);
        }
    }

    for (auto& t : threadArr) {
        t.join();
    }

}
