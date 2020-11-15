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
//#include "config.h"

#include <Python.h>

#include "tasks/imageHandler.h"

#include "dsHandler.h"
#include "singleton.h"
#include "config_yaml.h"

//#include "rtmpHandler.h"

using namespace std;

class Dispatch {
public:
    Dispatch();

    ~Dispatch();

    void run();

    //    ls add
    int mQueueArrLen = 2;
    int mQueueLen    = 5;

    //      total cam params
    vector<string>          mCamPath; // 拉流地址
    vector<string>          mRtmpPath; // 拉流地址
    vector<dsHandler *>     mDsHandlers;
    vector<cv::VideoWriter> mRTMPWriter; // 推流地址
    //        vector<rtmpHandler*> mRtmpHandlers; // 推流地址
    vector<bool>            mCamLive;

    vector<queue<cv::Mat> *> mQueueCam;
    vector<queue<cv::Mat>>   mQueue_rtmp;

    vector<condition_variable *> mCon_not_full;
    vector<condition_variable *> mCon_not_empty;
    vector<condition_variable *> mCon_rtmp;
    vector<mutex *>              mConMutexCam;
    vector<mutex *>              mConMutexRTMP;
    vector<mutex *>              mRtmpMutex;

    vector<imageHandler *>       mImageHandlers;
    vector<cv::Mat>              mRtmpImg;

    

private:

    int     frame_count;

    int frames_skip;
    int rtmp_mode;
    int IMV_mode; // 0 GPU 1 CPU
    void ProduceImage(int mode);

    void ConsumeImage(int mode);

    void ConsumeRTMPImage(int mode);

    void RPCServer();

    int addCam(bool isAdd, int cam_id, string rtsp_str, int is_h264);

    int removeCam(int cam_id);


    condition_variable vCon_not_full_0, vCon_not_full_1, vCon_not_full_2, vCon_not_full_3;
    condition_variable vCon_not_empty_0, vCon_not_empty_1, vCon_not_empty_2, vCon_not_empty_3;
    condition_variable vCon_rtmp_0, vCon_rtmp_1, vCon_rtmp_2, vCon_rtmp_3;
    mutex              vConMutexCam_0, vConMutexCam_1, vConMutexCam_2, vConMutexCam_3;
    mutex              vConMutexRTMP_0, vConMutexRTMP_1, vConMutexRTMP_2, vConMutexRTMP_3;
    mutex              vRtmpMutex_0, vRtmpMutex_1, vRtmpMutex_2, vRtmpMutex_3;
    cv::VideoWriter    writer_0, writer_1, writer_2, writer_3;
    dsHandler          *dsHandler_0, *dsHandler_1, *dsHandler_2, *dsHandler_3;
    //        rtmpHandler *rtmpHandler_0, *rtmpHandler_1, *rtmpHandler_2, *rtmpHandler_3;

    yamlConfig *config_A = Singleton<yamlConfig>::GetInstance("/srv/media_info.yaml");
};

#endif