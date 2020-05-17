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

    mQueueLen = stoi(labels["QLEN"]);
    frames_skip = stoi(labels["FRAMES_SKIP"]); //2
    rtmp_mode = stoi(labels["RTMP_MODE"]); //0
    int fps = stoi(labels["FPS"]);  //20
    int out_w = stoi(labels["OUT_W"]);  //2880
    int out_h = stoi(labels["OUT_H"]);  //960

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

    switch (mode) {
        case 0:
            lock = &myMutex_rtmp_front;
            queue = &mQueue_rtmp_front;
            con_v_wait = &con_rtmp_front;
            rtmpLock = &rtmpMutex_front;
//            rtmpHandler = &ls_handler_front;
            break;
        case 1:
            lock = &myMutex_rtmp_mid;
            queue = &mQueue_rtmp_mid;
            con_v_wait = &con_rtmp_mid;
            rtmpLock = &rtmpMutex_mid;
//            rtmpHandler = &ls_handler_mid;
            break;
        case 2:
            break;
        default:
            break;
    }


    while (true) {
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

    string path_0 = labels["video_path_0"];
    string path_1 = labels["video_path_1"];
    string fish_dat = labels["FISH_DAT"];
//    string path_0 = "/home/user/Program/ls-dev/dispatchProject/UnitTest/test_video/fish_0.mp4";
    cout << "path_0 " << path_0 << endl;

    cv::VideoCapture cam;
    cv::Mat frame = cv::imread(fish_dat);
    string path = "";
    mutex *lock;
    mutex *rtmpLock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait, *con_v_notification;
//    rtmpHandler* rtmpHandler;
    cv::Mat *rtmp_img;

    switch (mode){
        case 0:
            path = path_0;
            lock = &myMutex_front;
            queue = &mQueue_front;
            con_v_wait = &con_front_not_full;
            con_v_notification = &con_front_not_empty;
//            rtmpHandler = &ls_handler_front;
            rtmpLock = &rtmpMutex_front;
            rtmp_img = &rtmp_front_img;
            break;
        case 1:
            path = path_1;
            lock = &myMutex_mid;
            queue = &mQueue_mid;
            con_v_wait = &con_mid_not_full;
            con_v_notification = &con_mid_not_empty;
//            rtmpHandler = &ls_handler_mid;
            rtmpLock = &rtmpMutex_mid;
            rtmp_img = &rtmp_mid_img;
            break;
        case 2:
            break;
        default:
            break;
    }

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
    for (int i=0; ; i++) {

        cam.read(frame);

//        cout << "ProduceImage "<< mode << " img "<< i << " cost : " << getCurrentTime()-end << endl;
        end = getCurrentTime();

        if (i % frames_skip != 0 ){
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

    switch (mode){
        case 0:
            lock = &myMutex_front;
            queue = &mQueue_front;
            rtmpQueue = &mQueue_rtmp_front;
            con_v_wait = &con_front_not_empty;
            con_v_notification = &con_front_not_full;
            con_rtmp = &con_rtmp_front;
            rtmp_img = &rtmp_front_img;
            break;
        case 1:
            lock = &myMutex_mid;
            queue = &mQueue_mid;
            rtmpQueue = &mQueue_rtmp_mid;
            con_v_wait = &con_mid_not_empty;
            con_v_notification = &con_mid_not_full;
            con_rtmp = &con_rtmp_mid;
            rtmp_img = &rtmp_mid_img;
            break;
        case 2:
            break;
        default:
            break;
    }


    while (true){
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
    thread thread_write_image_front(&Dispatch::ProduceImage, this, 0);
    thread thread_write_image_mid(&Dispatch::ProduceImage, this, 1);

    thread thread_read_image_front(&Dispatch::ConsumeImage, this, 0);
    thread thread_read_image_mid(&Dispatch::ConsumeImage, this, 1);

    thread thread_RTMP_front(&Dispatch::ConsumeRTMPImage, this, 0);
    thread thread_RTMP_mid(&Dispatch::ConsumeRTMPImage, this, 1);

    thread thread_RPC_server(&Dispatch::RPCServer, this);

    thread_write_image_front.join();
    thread_write_image_mid.join();

    thread_read_image_front.join();
    thread_read_image_mid.join();

    thread_RTMP_front.join();
    thread_RTMP_mid.join();

    thread_RPC_server.join();


}
