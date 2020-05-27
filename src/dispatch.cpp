//
// Created by xinxueshi on 2020/4/28.
//  https://blog.csdn.net/yiyouxian/article/details/51993524
//
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <chrono>

#include "dispatch.h"
#include "config.h"
#include "Common.h"
#include "utils/vis.h"
#include "tasks/imageHandler.h"

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"



using namespace std;
using namespace cv;

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

    string path_0 = dbLabels["video_path_0"];
    string path_1 = dbLabels["video_path_1"];

    mCamLive = {path_0!="", path_1!="", false, false};

    mDsHandlers.resize(4);
    mCamLive.resize(4);
    mCamPath.resize(4);
    mRTMPWriter.resize(4);
//    mRtmpHandlers.resize(4);
    mQueueCam.resize(4);
    mQueue_rtmp.resize(4);

    mCon_not_full.resize(4);
    mCon_not_empty.resize(4);
    mCon_rtmp.resize(4);
    mConMutexCam.resize(4);
    mConMutexRTMP.resize(4);
    mRtmpMutex.resize(4);


    if (rtmp_mode){
        std::string rtmpPath_0 = labels["RTMP_PATH_0"]; //  rtmp://127.0.0.1:1935/hls/room
        std::string rtmpPath_1 = labels["RTMP_PATH_1"];
        std::string rtmpPath_2 = labels["RTMP_PATH_2"]; //  rtmp://127.0.0.1:1935/hls/room
        std::string rtmpPath_3 = labels["RTMP_PATH_3"];
        std::string GSTParams = "appsrc ! videoconvert ! nvvidconv ! nvv4l2h264enc ! h264parse ! queue ! flvmux ! rtmpsink location=";
//        std::string GSTParams = "appsrc ! videoconvert ! nvvidconv ! omxh264enc ! h264parse ! queue ! flvmux ! rtmpsink location=";

//        rtmpHandler_0 = new rtmpHandler("",rtmpPath_0,out_w,out_h,fps);
//        rtmpHandler_1 = new rtmpHandler("",rtmpPath_1,out_w,out_h,fps);
//        rtmpHandler_2 = new rtmpHandler("",rtmpPath_2,out_w,out_h,fps);
//        rtmpHandler_3 = new rtmpHandler("",rtmpPath_3,out_w,out_h,fps);

        rtmpPath_0 =  GSTParams + rtmpPath_0;
        rtmpPath_1 =  GSTParams + rtmpPath_1;
        rtmpPath_2 =  GSTParams + rtmpPath_2;
        rtmpPath_3 =  GSTParams + rtmpPath_3;

        writer_0 = cv::VideoWriter(rtmpPath_0,CAP_GSTREAMER,0, fps, cv::Size(out_w, out_h), true);
        writer_1 = cv::VideoWriter(rtmpPath_1,CAP_GSTREAMER,0, fps, cv::Size(out_w, out_h), true);
//        writer_2 = cv::VideoWriter(rtmpPath_2,CAP_GSTREAMER,0, fps, cv::Size(out_w, out_h), true);
//        writer_3 = cv::VideoWriter(rtmpPath_3,CAP_GSTREAMER,0, fps, cv::Size(out_w, out_h), true);

        mRTMPWriter = {writer_0, writer_1, writer_2, writer_3};
//        mRtmpHandlers = {rtmpHandler_0, rtmpHandler_1, rtmpHandler_2, rtmpHandler_3};
    }


    dsHandler_0 = new dsHandler (path_0,out_w,out_h,4000000, 0, 1, frames_skip);
    dsHandler_1 = new dsHandler (path_1,out_w,out_h,4000000, 1, 1, frames_skip);
    dsHandler_2 = new dsHandler();
    dsHandler_3 = new dsHandler();

    mCon_not_full  = { &vCon_not_full_0, &vCon_not_full_1, &vCon_not_full_2, &vCon_not_full_3 };
    mDsHandlers    = {dsHandler_0, dsHandler_1, dsHandler_2, dsHandler_3};
    mCon_not_empty = { &dsHandler_0->con_v_notification, &dsHandler_1->con_v_notification, &dsHandler_2->con_v_notification,
                       &dsHandler_3->con_v_notification};
    mCon_rtmp      = {&vCon_rtmp_0, &vCon_rtmp_1, &vCon_rtmp_2, &vCon_rtmp_3};
    mConMutexCam   = {&dsHandler_0->myMutex, &dsHandler_1->myMutex, &dsHandler_2->myMutex, &dsHandler_3->myMutex};
    mQueueCam      = {&dsHandler_0->imgQueue, &dsHandler_1->imgQueue, &dsHandler_2->imgQueue, &dsHandler_3->imgQueue};
    mConMutexRTMP  = {&vConMutexRTMP_0, &vConMutexRTMP_1, &vConMutexRTMP_2, &vConMutexRTMP_3};
    mRtmpMutex     = {&vRtmpMutex_0, &vRtmpMutex_1, &vRtmpMutex_2, &vRtmpMutex_3};



    mRtmpImg.resize(4);

    mCamPath[0] = path_0;
    mCamPath[1] = path_1;

    Py_Initialize();
    if (!Py_IsInitialized())
    {
        LOG_DEBUG("Py_Initialize error, return\n");
    }

    PyEval_InitThreads();
    int nInit = PyEval_ThreadsInitialized();
    if (nInit)
    {
        LOG_DEBUG("PyEval_SaveThread\n");
        PyEval_SaveThread();
    }

    cout << "end Init Dispatch" << endl;

}

Dispatch::~Dispatch() = default;

void Dispatch::RPCServer(){
    int out_w = stoi(labels["OUT_W"]);  //2880
    int out_h = stoi(labels["OUT_H"]);  //960

    vector<thread> threadArr;
    int socketPort = stoi(labels["SPORT"]);
    int socketQueue = stoi(labels["SQUEUE"]);

    int conn;
    int ss = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in server_sockaddr;
    server_sockaddr.sin_family = AF_INET;
    server_sockaddr.sin_port = htons(socketPort);
    server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if(bind(ss, (struct sockaddr* ) &server_sockaddr, sizeof(server_sockaddr))==-1) {
        perror("bind");
        exit(1);
    }
    if(listen(ss, socketQueue) == -1) {
        perror("listen");
        exit(1);
    }

    struct sockaddr_in client_addr;
    socklen_t length = sizeof(client_addr);

    char buffer[1024];
    char respBuffer[1024];

    printf("======waiting for client's request======\n");
    while(1) {
        ///成功返回非负描述字，出错返回-1
        conn = accept(ss, (struct sockaddr*)&client_addr, &length);
        if( conn < 0 ) {
            perror("connect");
            exit(1);
        }

        memset(buffer, 0 ,sizeof(buffer));
        memset(respBuffer, 0 ,sizeof(respBuffer));

        int len = recv(conn, buffer, sizeof(buffer), 0);
//        if(strcmp(buffer, "exit\n") == 0) break;
        printf("%s\n", buffer);

        string jsonStr = buffer;
        rapidjson::Document dom;
        string cmd_str, rtsp_str, mac_str, gs_rtsp_str;
        int cam_id, is_h264;
        bool isAdd;
        if (!dom.Parse(jsonStr.c_str()).HasParseError()) {
            if (dom.HasMember("cmd") && dom["cmd"].IsString()) {
                cmd_str = dom["cmd"].GetString();
                std::cout << cmd_str<< std::endl;
            }
            if (dom.HasMember("cam_id") && dom["cam_id"].IsInt()) {
                cam_id = dom["cam_id"].GetInt();
                std::cout << cam_id<< std::endl;
            }
            if (dom.HasMember("rtsp") && dom["rtsp"].IsString()) {
                rtsp_str = dom["rtsp"].GetString();
                std::cout << rtsp_str<< std::endl;
            }
            if (dom.HasMember("mac") && dom["mac"].IsString()) {
                mac_str = dom["mac"].GetString();
                std::cout << mac_str<< std::endl;
            }
            if (dom.HasMember("is_h264") && dom["is_h264"].IsInt()) {
                is_h264 = dom["is_h264"].GetInt();
                std::cout << is_h264<< std::endl;
            }


        }else{
            printf("fail to parse json str\n");
        }

        rapidjson::StringBuffer buf;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buf);

        writer.StartObject();
        writer.Key("message"); writer.String("OK");
        writer.Key("code"); writer.Int(0);
        writer.EndObject();
        const char* json_content = buf.GetString();
        strcpy(respBuffer, json_content);

        //必须要有返回数据， 这样才算一个完整的请求
        send(conn, respBuffer, strlen(respBuffer) , 0);
        close(conn);

        string add = "add";
        isAdd = add.compare(cmd_str) == 0;

        if (isAdd){
            if (!mCamLive[cam_id]){
                mCamLive[cam_id] = isAdd;
                mCamPath[cam_id] = rtsp_str;
                cout << isAdd << endl;
                cout << "socket Add cam " << cam_id << endl ;
                switch (cam_id){
                    case 0:
                        dsHandler_0 = new dsHandler (rtsp_str,out_w,out_h, 4000000, cam_id, is_h264 ? 0 : 1, frames_skip);

                        mDsHandlers[cam_id] = dsHandler_0;
                        mCon_not_empty[cam_id] = &dsHandler_0->con_v_notification;
                        mConMutexCam[cam_id] = &dsHandler_0->myMutex;
                        mQueueCam[cam_id] = &dsHandler_0->imgQueue;
                        break;
                    case 1:
                        dsHandler_1 = new dsHandler (rtsp_str,out_w,out_h, 4000000, cam_id, is_h264 ? 0 : 1, frames_skip);

                        mDsHandlers[cam_id] = dsHandler_1;
                        mCon_not_empty[cam_id] = &dsHandler_1->con_v_notification;
                        mConMutexCam[cam_id] = &dsHandler_1->myMutex;
                        mQueueCam[cam_id] = &dsHandler_1->imgQueue;
                        break;
                    case 2:
                        dsHandler_2 = new dsHandler (rtsp_str,out_w,out_h, 4000000, cam_id, is_h264 ? 0 : 1, frames_skip);

                        mDsHandlers[cam_id] = dsHandler_2;
                        mCon_not_empty[cam_id] = &dsHandler_2->con_v_notification;
                        mConMutexCam[cam_id] = &dsHandler_2->myMutex;
                        mQueueCam[cam_id] = &dsHandler_2->imgQueue;
                        break;
                    case 3:
                        dsHandler_3 = new dsHandler (rtsp_str,out_w,out_h, 4000000, cam_id, is_h264 ? 0 : 1, frames_skip);

                        mDsHandlers[cam_id] = dsHandler_3;
                        mCon_not_empty[cam_id] = &dsHandler_3->con_v_notification;
                        mConMutexCam[cam_id] = &dsHandler_3->myMutex;
                        mQueueCam[cam_id] = &dsHandler_3->imgQueue;
                        break;
                    default:
                        break;
                }

                threadArr.emplace_back(&Dispatch::ProduceImage, this, cam_id);
                threadArr.emplace_back(&Dispatch::ConsumeImage, this, cam_id);
                threadArr.emplace_back(&Dispatch::ConsumeRTMPImage, this, cam_id);
            }

        } else{
            cout << "socket Del cam " << cam_id << endl ;
            mCamLive[cam_id] = false;
            mDsHandlers[cam_id] ->finish();
            mDsHandlers[cam_id] = nullptr;
            cout << "del cam_id : " << cam_id << endl;
        }


        cout << gs_rtsp_str <<endl;

        std::cout << "*********over***********"<<endl;
    }
    close(ss);

}

void Dispatch::ConsumeRTMPImage(int mode){

    cv::Mat img;

    int num = 0;

    mutex *lock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait;
    mutex* rtmpLock;
    cv::VideoWriter writer;
//    rtmpHandler* vRtmpHandler;

    lock = mConMutexRTMP[mode];
    queue = &mQueue_rtmp[mode];
    con_v_wait = mCon_rtmp[mode];
    rtmpLock = mRtmpMutex[mode];
    writer = mRTMPWriter[mode];
//    vRtmpHandler = mRtmpHandlers[mode];

    cout << "ConsumeRTMPImage  start " << endl;
    cout << lock << endl;
    cout << con_v_wait << endl;
    cout << rtmpLock << endl;
    cout << "ConsumeRTMPImage  end " << endl;

    while (mCamLive[mode]) {
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->empty()) {
            con_v_wait->wait(guard);
        }

        img = queue->front();
        queue->pop();
        guard.unlock();

        rtmpLock->lock();
//        TODO 推流逻辑
//        vRtmpHandler->pushRTMP(img);
        writer.write(img);
        rtmpLock->unlock();
        num++;
        if (num == 10000) num = 0;
    }

    cout << "ConsumeRTMPImage finish "<< mode << endl;

}

void Dispatch::ProduceImage(int mode){
    cout << "produceImage "<< mode << " start" << endl;
    dsHandler* mDsHandler = mDsHandlers[mode];
    mDsHandler->run();
    cout << "produceImage "<< mode << " finish" << endl;

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

    imageHandler vImageHandler;

    lock = mConMutexCam[mode];
    queue = mQueueCam[mode];
    rtmpQueue = &mQueue_rtmp[mode];
    con_v_wait = mCon_not_empty[mode];
    con_v_notification = mCon_not_full[mode];
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
            con_v_wait->wait(guard);
        }

        frame = queue->front();
        queue->pop();
//        con_v_notification->notify_all();
        guard.unlock();

        //        TODO 业务逻辑
        ret_img = frame.clone();
        if (stoi(labels["inference_switch"])){
            vImageHandler.run(ret_img);
            vImageHandler.vis(ret_img);
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

        num++;
        if (num == 10000) num = 0;
    }

    cout << " ConsumeImage finish "<< mode << endl;

}

void Dispatch::run(){

    vector<thread> threadArr;
    threadArr.emplace_back(&Dispatch::RPCServer, this);

    for (int i = 0; i < mCamLive.size(); ++i) {
        if (mCamLive[i]){
            cout << "start woker thread " << endl;
            threadArr.emplace_back(&Dispatch::ProduceImage, this, i);
            threadArr.emplace_back(&Dispatch::ConsumeImage, this, i);
            threadArr.emplace_back(&Dispatch::ConsumeRTMPImage, this, i);

            this_thread::sleep_for(chrono::seconds(10));
        }
    }

    for (auto& t : threadArr) {
        t.join();
    }

}
