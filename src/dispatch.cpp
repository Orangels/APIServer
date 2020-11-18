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

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"


using namespace std;
using namespace cv;

int64_t getCurrentTime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

Dispatch::Dispatch(){
    cout << "start Init Dispatch" << endl;

    mDsHandlers.resize(4);
    //    mCamLive.resize(4);
    mCamPath.resize(4);
    mRTMPWriter.resize(4);
    mQueueCam.resize(4);
    mQueue_rtmp.resize(4);

    mCon_not_full.resize(4);
    mCon_not_empty.resize(4);
    mCon_rtmp.resize(4);
    mConMutexCam.resize(4);
    mConMutexRTMP.resize(4);
    mRtmpMutex.resize(4);
    mRtmpPath.resize(4);

    mImageHandlers.resize(4);
    mCamLive = {false, false, false, false};

    auto conf = config_A->getConfig();

    //    yaml Config
    if (conf["CAM"].size() > 0) {
        if (conf["CAM"][0].Type() == 1) {
            mQueueLen   = 20;
            frames_skip = 2;
            rtmp_mode   = 1;
        } else {
            mQueueLen   = conf["CAM"][0]["QLEN"].as<int>();
            frames_skip = conf["CAM"][0]["FRAMES_SKIP"].as<int>();
            rtmp_mode   = conf["CAM"][0]["RTMP_MODE"].as<int>();
        }


        //        两个循环是因为推流要在拉流之前初始化
        for (int i = 0; i < conf["CAM"].size(); ++i) {
            cout << "初始化 writer -- " << i << endl;
            if (conf["CAM"][i].Type() == 1) {
                cout << "cam type -- " << 1 << endl;

                //                空值, 初始化默认
                int fps   = 10;
                int out_w = 1280;
                int out_h = 720;

                mRtmpPath[i] = "rtmp://127.0.0.1:1935/hls/00" + to_string(i);
                string GSTParams = "appsrc ! videoconvert ! nvvidconv ! nvv4l2h264enc ! h264parse ! queue ! flvmux ! rtmpsink location=";

                mRtmpPath[i]   = GSTParams + mRtmpPath[i];
                mRTMPWriter[i] = cv::VideoWriter(mRtmpPath[i], CAP_GSTREAMER, 0, fps, cv::Size(out_w, out_h), true);
                continue;

            }

            int fps   = conf["CAM"][i]["FPS"].as<int>();
            int out_w = conf["CAM"][i]["CAMERA_TYPE"]["RTMP_SIZE"]["WIDTH"].as<int>();
            int out_h = conf["CAM"][i]["CAMERA_TYPE"]["RTMP_SIZE"]["HEIGHT"].as<int>();

            mRtmpPath[i] = conf["CAM"][i]["RTMP_PATH"][0].as<string>();
            string GSTParams = "appsrc ! videoconvert ! nvvidconv ! nvv4l2h264enc ! h264parse ! queue ! flvmux ! rtmpsink location=";

            mRtmpPath[i]   = GSTParams + mRtmpPath[i];
            mRTMPWriter[i] = cv::VideoWriter(mRtmpPath[i], CAP_GSTREAMER, 0, fps, cv::Size(out_w, out_h), true);

            cout << " rtmp path = " << mRtmpPath[i] << endl;
        }

        for (int i = 0; i < conf["CAM"].size(); ++i) {
            cout << "初始化 dsHandler -- " << i << endl;
            if (conf["CAM"][i].Type() == 1) {
                //                空值
                cout << "cam type -- " << 1 << endl;
                continue;
            }

            mCamLive[i] = true;

            int    path_h264   = conf["CAM"][i]["CAMERA_TYPE"]["FORMAT_H264"].as<bool>() == false ? 1 : 0;
            string stream_path = conf["CAM"][i]["VIDEO_PATH"].as<string>();

            mCamPath[i]       = stream_path;
            mDsHandlers[i]    = new dsHandler(stream_path, 1280, 720, 4000000, i, path_h264, frames_skip);
            mCon_not_empty[i] = &mDsHandlers[i]->mCon_not_empty;
            mCon_not_full[i]  = &mDsHandlers[i]->mCon_not_full;
            mConMutexCam[i]   = &mDsHandlers[i]->myMutex;
            mQueueCam[i]      = &mDsHandlers[i]->imgQueue;

        }


    }

    //    mCon_not_full = {&vCon_not_full_0, &vCon_not_full_1, &vCon_not_full_2, &vCon_not_full_3};
    mCon_rtmp     = {&vCon_rtmp_0, &vCon_rtmp_1, &vCon_rtmp_2, &vCon_rtmp_3};
    mConMutexRTMP = {&vConMutexRTMP_0, &vConMutexRTMP_1, &vConMutexRTMP_2, &vConMutexRTMP_3};
    mRtmpMutex    = {&vRtmpMutex_0, &vRtmpMutex_1, &vRtmpMutex_2, &vRtmpMutex_3};

    mRtmpImg.resize(4);


    Py_Initialize();
    if (!Py_IsInitialized()) {
        LOG_DEBUG("Py_Initialize error, return\n");
    }

    PyEval_InitThreads();
    int nInit = PyEval_ThreadsInitialized();
    if (nInit) {
        LOG_DEBUG("PyEval_SaveThread\n");
        PyEval_SaveThread();
    }

    cout << "end Init Dispatch" << endl;

}

Dispatch::~Dispatch() = default;

int Dispatch::addCam(bool isAdd, int cam_id, string rtsp_str, int is_h264){
    if (!mCamLive[cam_id]) {
        mCamLive[cam_id] = isAdd;
        mCamPath[cam_id] = rtsp_str;
        cout << isAdd << endl;
        cout << "socket Add cam " << cam_id << endl;

        mDsHandlers[cam_id]    = new dsHandler(rtsp_str, 1280, 720, 4000000, cam_id, is_h264 ? 0 : 1, frames_skip);
        mCon_not_empty[cam_id] = &mDsHandlers[cam_id]->mCon_not_empty;
        mConMutexCam[cam_id]   = &mDsHandlers[cam_id]->myMutex;
        mQueueCam[cam_id]      = &mDsHandlers[cam_id]->imgQueue;
    }
}

int Dispatch::removeCam(int cam_id){
    cout << "socket Del cam " << cam_id << endl;
    mCamLive[cam_id] = false;
    mDsHandlers[cam_id]->finish();
    mDsHandlers[cam_id] = nullptr;
    cout << "del cam_id : " << cam_id << endl;
}

void Dispatch::RPCServer(){
    vector<thread> threadArr;

    auto conf        = config_A->getConfig();
    int  socketPort  = conf["SERVER"]["SPORT"].as<int>();
    int  socketQueue = conf["SERVER"]["SQUEUE"].as<int>();

    int conn;
    int ss           = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in server_sockaddr;
    server_sockaddr.sin_family      = AF_INET;
    server_sockaddr.sin_port        = htons(socketPort);
    server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(ss, (struct sockaddr *) &server_sockaddr, sizeof(server_sockaddr)) == -1) {
        perror("bind");
        exit(1);
    }
    if (listen(ss, socketQueue) == -1) {
        perror("listen");
        exit(1);
    }

    struct sockaddr_in client_addr;
    socklen_t          length       = sizeof(client_addr);

    char buffer[1024];
    char respBuffer[1024];

    printf("======waiting for client's request======\n");
    while (1) {
        ///成功返回非负描述字，出错返回-1
        conn = accept(ss, (struct sockaddr *) &client_addr, &length);
        if (conn < 0) {
            perror("connect");
            exit(1);
        }

        memset(buffer, 0, sizeof(buffer));
        memset(respBuffer, 0, sizeof(respBuffer));

        int len = recv(conn, buffer, sizeof(buffer), 0);
        //        if(strcmp(buffer, "exit\n") == 0) break;
        printf("%s\n", buffer);

        string              jsonStr = buffer;
        rapidjson::Document dom;
        string              cmd_str, rtsp_str, mac_str, gs_rtsp_str, config_key;
        int                 cam_id, is_h264;
        bool                isAdd;

        if (!dom.Parse(jsonStr.c_str()).HasParseError()) {
            if (dom.HasMember("cmd") && dom["cmd"].IsString()) {
                cmd_str = dom["cmd"].GetString();
                std::cout << cmd_str << std::endl;
            }
            if (dom.HasMember("cam_id") && dom["cam_id"].IsInt()) {
                cam_id = dom["cam_id"].GetInt();
                std::cout << cam_id << std::endl;
            }
            if (dom.HasMember("rtsp") && dom["rtsp"].IsString()) {
                rtsp_str = dom["rtsp"].GetString();
                std::cout << rtsp_str << std::endl;
            }
            if (dom.HasMember("mac") && dom["mac"].IsString()) {
                mac_str = dom["mac"].GetString();
                std::cout << mac_str << std::endl;
            }
            if (dom.HasMember("is_h264") && dom["is_h264"].IsInt()) {
                is_h264 = dom["is_h264"].GetInt();
                std::cout << is_h264 << std::endl;
            }
            if (dom.HasMember("mode") && dom["mode"].IsString()) {
                config_key = dom["mode"].GetString();
                std::cout << config_key << std::endl;

            }


        } else {
            printf("fail to parse json str\n");
        }

        rapidjson::StringBuffer                           buf;
        rapidjson::PrettyWriter <rapidjson::StringBuffer> writer(buf);

        writer.StartObject();
        writer.Key("message");
        writer.String("OK");
        writer.Key("code");
        writer.Int(0);
        writer.EndObject();
        const char *json_content = buf.GetString();
        strcpy(respBuffer, json_content);

        //必须要有返回数据， 这样才算一个完整的请求
        send(conn, respBuffer, strlen(respBuffer), 0);
        close(conn);

        string add = "add";
        isAdd = add.compare(cmd_str) == 0;

        if (isAdd) {

            //            addCam(isAdd, cam_id, rtsp_str, is_h264);
            //            threadArr.emplace_back(&Dispatch::ProduceImage, this, cam_id);
            //            threadArr.emplace_back(&Dispatch::ConsumeImage, this, cam_id);
            //            threadArr.emplace_back(&Dispatch::ConsumeRTMPImage, this, cam_id);

            if (!mCamLive[cam_id]) {
                mCamLive[cam_id] = isAdd;
                mCamPath[cam_id] = rtsp_str;
                cout << isAdd << endl;
                cout << "socket Add cam " << cam_id << endl;
                switch (cam_id) {
                    case 0:
                        dsHandler_0 = new dsHandler(rtsp_str, 1280, 720, 4000000, cam_id, is_h264 ? 0 : 1,
                                                    frames_skip);

                        mDsHandlers[cam_id]    = dsHandler_0;
                        mCon_not_empty[cam_id] = &dsHandler_0->mCon_not_empty;
                        mConMutexCam[cam_id]   = &dsHandler_0->myMutex;
                        mQueueCam[cam_id]      = &dsHandler_0->imgQueue;
                        break;
                    case 1:
                        dsHandler_1 = new dsHandler(rtsp_str, 1280, 720, 4000000, cam_id, is_h264 ? 0 : 1,
                                                    frames_skip);

                        mDsHandlers[cam_id]    = dsHandler_1;
                        mCon_not_empty[cam_id] = &dsHandler_1->mCon_not_empty;
                        mConMutexCam[cam_id]   = &dsHandler_1->myMutex;
                        mQueueCam[cam_id]      = &dsHandler_1->imgQueue;
                        break;
                    case 2:
                        dsHandler_2 = new dsHandler(rtsp_str, 1280, 720, 4000000, cam_id, is_h264 ? 0 : 1,
                                                    frames_skip);

                        mDsHandlers[cam_id]    = dsHandler_2;
                        mCon_not_empty[cam_id] = &dsHandler_2->mCon_not_empty;
                        mConMutexCam[cam_id]   = &dsHandler_2->myMutex;
                        mQueueCam[cam_id]      = &dsHandler_2->imgQueue;
                        break;
                    case 3:
                        dsHandler_3 = new dsHandler(rtsp_str, 1280, 720, 4000000, cam_id, is_h264 ? 0 : 1,
                                                    frames_skip);

                        mDsHandlers[cam_id]    = dsHandler_3;
                        mCon_not_empty[cam_id] = &dsHandler_3->mCon_not_empty;
                        mConMutexCam[cam_id]   = &dsHandler_3->myMutex;
                        mQueueCam[cam_id]      = &dsHandler_3->imgQueue;
                        break;
                    default:
                        break;
                }

                threadArr.emplace_back(&Dispatch::ProduceImage, this, cam_id);
                threadArr.emplace_back(&Dispatch::ConsumeImage, this, cam_id);
                threadArr.emplace_back(&Dispatch::ConsumeRTMPImage, this, cam_id);
            }

        } else if (string("set").compare(cmd_str) == 0) {
            //            修改
            const rapidjson::Value &params = dom["params"];
            cout << "获取参数" << endl;

            if (params.HasMember("ALGORITHM") && params["ALGORITHM"].IsObject()) {
                //                修改rtmp 尺寸
                int LOS_NUMBER = params["ALGORITHM"]["TRACKER"]["LOS_NUMBER"].GetInt();
                cout << "修改 LOS_NUMBER " << endl;
                mutex rtmpLock;
                rtmpLock.lock();
                mImageHandlers[cam_id]->updateLosNum(LOS_NUMBER);
                rtmpLock.unlock();
                cout << "修改 LOS_NUMBER 完成" << endl;

            } else {
                cout << "参数错误" << endl;
            }

        } else {
            //            removeCam(cam_id);
            //            删除
            cout << "socket Del cam " << cam_id << endl;
            mCamLive[cam_id] = false;
            //            this_thread::sleep_for(chrono::milliseconds(200));
//            this_thread::sleep_for(chrono::seconds(3));
            cout << "produce pre del" << endl;
            mDsHandlers[cam_id]->finish();
            //            在 produce 线程做结束处理
            //            mDsHandlers[cam_id] = nullptr;
            cout << "del cam_id : " << cam_id << endl;
        }


        cout << gs_rtsp_str << endl;

        std::cout << "*********over***********" << endl;
    }
    close(ss);

}

void Dispatch::ConsumeRTMPImage(int mode){

    cv::Mat img;

    int num = 0;

    auto conf  = config_A->getConfig();
    int  fps   = conf["CAM"][mode]["FPS"].as<int>();
    int  out_w = conf["CAM"][mode]["CAMERA_TYPE"]["RTMP_SIZE"]["WIDTH"].as<int>();
    int  out_h = conf["CAM"][mode]["CAMERA_TYPE"]["RTMP_SIZE"]["HEIGHT"].as<int>();

    mutex              *lock;
    queue <cv::Mat>    *queue;
    condition_variable *con_v_wait;
    mutex              *rtmpLock;

    cv::VideoWriter writer;

    lock       = mConMutexRTMP[mode];
    queue      = &mQueue_rtmp[mode];
    con_v_wait = mCon_rtmp[mode];
    rtmpLock   = mRtmpMutex[mode];
    writer     = mRTMPWriter[mode];
    //    mRTMPWriter[mode] = writer;

    cout << "ConsumeRTMPImage  start " << endl;
    cout << lock << endl;
    cout << con_v_wait << endl;
    cout << rtmpLock << endl;
    cout << "ConsumeRTMPImage  end " << endl;

    while (mCamLive[mode]) {
        std::unique_lock<std::mutex> guard(*lock);
        while (queue->empty()) {
            con_v_wait->wait(guard);
        }

        img = queue->front().clone();
        queue->pop();
        guard.unlock();

        rtmpLock->lock();
        //        TODO 推流逻辑

        cv::resize(img, img, cv::Size(out_w, out_h));

        writer.write(img);
        writer.write(img);
        rtmpLock->unlock();
        num++;
        if (num == 10000) num = 0;
    }

    cout << "ConsumeRTMPImage finish " << mode << endl;

}

void Dispatch::ProduceImage(int mode){
    cout << "produceImage " << mode << " start" << endl;
    dsHandler *mDsHandler = mDsHandlers[mode];
    mDsHandler->run();
    cout << "produceImage " << mode << " finish" << endl;
    mCamLive[mode]    = false;
    //    delete mDsHandlers[mode];
    mDsHandlers[mode] = nullptr;

}


void Dispatch::ConsumeImage(int mode){

    auto conf             = config_A->getConfig();
    int  out_w            = conf["CAM"][mode]["CAMERA_TYPE"]["RTMP_SIZE"]["WIDTH"].as<int>();
    int  out_h            = conf["CAM"][mode]["CAMERA_TYPE"]["RTMP_SIZE"]["HEIGHT"].as<int>();
    bool inference_switch = conf["CAM"][mode]["INFERENCE_SWITCH"].as<bool>();

    int num = 0;

    cv::Mat frame;

    mutex              *lock;
    queue <cv::Mat>    *rtmpQueue;
    queue <cv::Mat>    *queue;
    condition_variable *con_rtmp;
    condition_variable *con_v_wait, *con_v_notification;
    cv::Mat            *rtmp_img;
    cv::Mat            ret_img;

    imageHandler *vImageHandler = new imageHandler(mode);
    mImageHandlers[mode] = vImageHandler;

    lock               = mConMutexCam[mode];
    queue              = mQueueCam[mode];
    rtmpQueue          = &mQueue_rtmp[mode];
    con_v_wait         = mCon_not_empty[mode];
    con_v_notification = mCon_not_full[mode];
    con_rtmp           = mCon_rtmp[mode];
    rtmp_img           = &mRtmpImg[mode];

    cout << "ConsumeImage  start " << endl;
    cout << lock << endl;
    cout << con_v_wait << endl;
    cout << con_v_notification << endl;
    cout << con_rtmp << endl;
    cout << "ConsumeImage  end " << endl;


    while (mCamLive[mode]) {

        int64_t start_time = getCurrentTime();

        std::unique_lock<std::mutex> guard(*lock);
        while (queue->empty()) {
            con_v_wait->wait(guard);
        }
        //        int64_t start_time = getCurrentTime();
        frame = queue->front();
        queue->pop();
        con_v_notification->notify_all();
        guard.unlock();

        //        TODO 业务逻辑
        ret_img = frame.clone();
        if (inference_switch) {
            //            cout << "mode -- " << mode << endl;
            mImageHandlers[mode]->run(ret_img, num);
            mImageHandlers[mode]->vis(ret_img);
        }

        if (rtmp_mode == 1) {
            // 推流
            cv::Mat rtmp_frame;
            cv::resize(ret_img, rtmp_frame, cv::Size(out_w, out_h));
            cv::Mat frame_clone = rtmp_frame.clone();
            rtmpQueue->push(frame_clone);
            con_rtmp->notify_all();
            //            *rtmp_img = frame_clone;
            //            *rtmp_img = ret_img;
            //            rtmpQueue->push(*rtmp_img);
            //            rtmpQueue->push(ret_img);
            //            con_rtmp->notify_all();
        }

        int64_t end_time = getCurrentTime();

//        cout << "mode " << mode << " num " << num << " total cost -- " << end_time - start_time << endl;

        num++;
//        if (num == 10000) num = 0;
    }


    //结束线程, 清理缓存
    std::unique_lock<std::mutex> guard(*lock);
    for (int i = 0; i < queue->size(); ++i) {
        queue->pop();
        con_v_notification->notify_all();
    }
    guard.unlock();

    cout << " ConsumeImage finish " << mode << endl;
//    这里删除 py 调用会有问题, 暂时不删除
//    delete mImageHandlers[mode];

}

void Dispatch::run(){

    vector<thread> threadArr;
    threadArr.emplace_back(&Dispatch::RPCServer, this);

    for (int i = 0; i < mCamLive.size(); ++i) {
        if (mCamLive[i]) {
            cout << "start woker thread " << endl;
            threadArr.emplace_back(&Dispatch::ConsumeImage, this, i);
            threadArr.emplace_back(&Dispatch::ConsumeRTMPImage, this, i);

            this_thread::sleep_for(chrono::seconds(10));
            threadArr.emplace_back(&Dispatch::ProduceImage, this, i);
        }
    }

    for (auto &t : threadArr) {
        t.join();
    }

}
