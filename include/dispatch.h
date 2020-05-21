//
// Created by xinxueshi on 2020/4/28.
//
#ifndef DISPATCH_H
#define DISPATCH_H

#include <iostream>

//ls add
#include <opencv2/core.hpp>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>
#include "config.h"
#include "utils/track.h"
#include "utils/match_id.h"
#include <Python.h>

#include "detection.h"

using namespace std;
class Dispatch
{
    public:
        Dispatch();
        ~Dispatch();
        void run();
        void test();
        int engine_Test();

        //    ls add
        void multithreadTest();
        int mQueueArrLen = 2;
        int mQueueLen = 5;

//      total cam params
        vector<string> mCamPath; // 拉流地址
        vector<cv::VideoWriter> mRTMPWriter; // 推流地址
        vector<bool> mCamLive;

        vector<queue<cv::Mat>> mQueueCam;
        vector<queue<cv::Mat>> mQueue_rtmp;

        vector<condition_variable*> mCon_not_full;
        vector<condition_variable*> mCon_not_empty;
        vector<condition_variable*> mCon_rtmp;
        vector<mutex*> mConMutexCam;
        vector<mutex*> mConMutexRTMP;
        vector<mutex*> mRtmpMutex;

        vector<cv::Mat> mRtmpImg;

//        vector<SSD_Detection> mSSD_Detections;


//        front cam params
        queue<cv::Mat> mQueue_front;
        queue<cv::Mat> mQueue_rtmp_front;

        condition_variable con_front_not_full;
        condition_variable con_front_not_empty;

        condition_variable con_rtmp_front;

        mutex myMutex_front;
        mutex myMutex_rtmp_front;

        mutex rtmpMutex_front;

        cv::Mat rtmp_front_img;

        bool camera_front;

    //        middle cam params
        queue<cv::Mat> mQueue_mid;
        queue<cv::Mat> mQueue_rtmp_mid;

        condition_variable con_mid_not_full;
        condition_variable con_mid_not_empty;

        condition_variable con_rtmp_mid;

        mutex myMutex_mid;
        mutex myMutex_rtmp_mid;

        mutex rtmpMutex_mid;

        cv::Mat rtmp_mid_img;

        bool camera_mid;

//        rpc params


        Match_ID* matcher;

private:
        Cconfig labels = Cconfig("../cfg/process.ini");
        int frame_count;

        int frames_skip;
        int rtmp_mode;
        int IMV_mode; // 0 GPU 1 CPU
        void ProduceImage(int mode);
        void ConsumeImage(int mode);
        void ConsumeRTMPImage(int mode);
        void RPCServer();


    condition_variable vCon_not_full_0, vCon_not_full_1, vCon_not_full_2, vCon_not_full_3;
    condition_variable vCon_not_empty_0, vCon_not_empty_1, vCon_not_empty_2, vCon_not_empty_3;
    condition_variable vCon_rtmp_0, vCon_rtmp_1, vCon_rtmp_2, vCon_rtmp_3;
    mutex vConMutexCam_0, vConMutexCam_1, vConMutexCam_2, vConMutexCam_3;
    mutex vConMutexRTMP_0, vConMutexRTMP_1, vConMutexRTMP_2, vConMutexRTMP_3;
    mutex vRtmpMutex_0, vRtmpMutex_1, vRtmpMutex_2, vRtmpMutex_3;
    cv::VideoWriter writer_0, writer_1, writer_2, writer_3;
//    SSD_Detection ssd_detection_0, ssd_detection_1;
//        Engine_api * pyEngineAPI;
//
//        Engine_api * pyEngineAPI_0;
//        Engine_api * pyEngineAPI_1;
//        Engine_api * pyEngineAPI_2;


};
#endif