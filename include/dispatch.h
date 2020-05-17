//
// Created by xinxueshi on 2020/4/28.
//
#ifndef DISPATCH_H
#define DISPATCH_H

#include <iostream>

//ls add
#include <opencv2/core.hpp>
#include <libgearman/gearman.h>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>
#include "rtmpHandler.h"
#include "config.h"
#include "utils/track.h"
#include "utils/match_id.h"
#include <Python.h>
#include "EnginePy.hpp"

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

//        front cam params
//        queue<cv::Mat> mQueue_front;
        queue<vector<cv::Mat>> mQueue_front;
        queue<cv::Mat> mQueue_rtmp_front;

        condition_variable con_front_not_full;
        condition_variable con_front_not_empty;

        condition_variable con_rtmp_front;

        mutex myMutex_front;
        mutex myMutex_rtmp_front;

        mutex rtmpMutex_front;

        rtmpHandler ls_handler_front;
        cv::Mat rtmp_front_img;

        bool camera_front;

    //        middle cam params
        queue<vector<cv::Mat>> mQueue_mid;
        queue<cv::Mat> mQueue_rtmp_mid;

        condition_variable con_mid_not_full;
        condition_variable con_mid_not_empty;

        condition_variable con_rtmp_mid;

        mutex myMutex_mid;
        mutex myMutex_rtmp_mid;

        mutex rtmpMutex_mid;

        rtmpHandler ls_handler_mid;
        cv::Mat rtmp_mid_img;

        bool camera_mid;

//        rpc params
        gearman_return_t gearRet;
        gearman_return_t gearWRet;
        gearman_client_st* gearClient;
        gearman_worker_st* gearWorker;


//        PyObject* pIns;
//        PyObject* pIns1;
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

        Engine_api * pyEngineAPI;

        Engine_api * pyEngineAPI_0;
        Engine_api * pyEngineAPI_1;
        Engine_api * pyEngineAPI_2;


};
#endif