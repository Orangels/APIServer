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

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"



using namespace std;
using namespace cv;

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
    mRTMPWriter.resize(4);
    mQueueCam.resize(4);
    mQueue_rtmp.resize(4);

    mCon_not_full.resize(4);
    mCon_not_empty.resize(4);
    mCon_rtmp.resize(4);
    mConMutexCam.resize(4);
    mConMutexRTMP.resize(4);
    mRtmpMutex.resize(4);

//    mSSD_Detections.resize(4);

    mCamLive = {true, true, false, false};

    mCon_not_full = { &vCon_not_full_0, &vCon_not_full_1, &vCon_not_full_2, &vCon_not_full_3 };
    mCon_not_empty = { &vCon_not_empty_0, &vCon_not_empty_1, &vCon_not_empty_2, &vCon_not_empty_3};
    mCon_rtmp = {&vCon_rtmp_0, &vCon_rtmp_1, &vCon_rtmp_2, &vCon_rtmp_3};
    mConMutexCam = {&vConMutexCam_0, &vConMutexCam_1, &vConMutexCam_2, &vConMutexCam_3};
    mConMutexRTMP = {&vConMutexRTMP_0, &vConMutexRTMP_1, &vConMutexRTMP_2, &vConMutexRTMP_3};
    mRtmpMutex = {&vRtmpMutex_0, &vRtmpMutex_1, &vRtmpMutex_2, &vRtmpMutex_3};

    SSD_Detection ssd_detection_0,ssd_detection_1;
//    mSSD_Detections = {ssd_detection_0, ssd_detection_1};

    mRtmpImg.resize(4);

    mCamPath[0] = path_0;
    mCamPath[1] = path_1;



    if (rtmp_mode){
        std::string rtmpPath_0 = labels["RTMP_PATH_0"]; //  rtmp://127.0.0.1:1935/hls/room
        std::string rtmpPath_1 = labels["RTMP_PATH_1"];
        std::string rtmpPath_2 = labels["RTMP_PATH_2"]; //  rtmp://127.0.0.1:1935/hls/room
        std::string rtmpPath_3 = labels["RTMP_PATH_3"];
        cout << rtmpPath_0 << endl;
        cout << rtmpPath_1 << endl;

        writer_0 = cv::VideoWriter(rtmpPath_0,CAP_GSTREAMER,0, fps, cv::Size(out_w, out_h), true);
        writer_1 = cv::VideoWriter(rtmpPath_1,CAP_GSTREAMER,0, fps, cv::Size(out_w, out_h), true);
        writer_2 = cv::VideoWriter(rtmpPath_2,CAP_GSTREAMER,0, fps, cv::Size(out_w, out_h), true);
        writer_3 = cv::VideoWriter(rtmpPath_3,CAP_GSTREAMER,0, fps, cv::Size(out_w, out_h), true);

        if (!writer_0.isOpened()) {
            std::cerr<< "0 Unable to open video file for writing" << std::endl;
            return ;
        }

        if (!writer_1.isOpened()) {
            std::cerr<< "1 Unable to open video file for writing" << std::endl;
            return ;
        }

        mRTMPWriter = {writer_0, writer_1, writer_2, writer_3};
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

        if (is_h264){
            gs_rtsp_str = "rtspsrc location=" + rtsp_str + " latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! appsink";
        } else{
            gs_rtsp_str = "rtspsrc location=" + rtsp_str + " latency=0 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! appsink";
        }
        string add = "add";
        isAdd = add.compare(cmd_str) == 0;

        mCamLive[cam_id] = isAdd;
        mCamPath[cam_id] = gs_rtsp_str;

        threadArr.emplace_back(&Dispatch::ProduceImage, this, cam_id);
        threadArr.emplace_back(&Dispatch::ConsumeImage, this, cam_id);
        threadArr.emplace_back(&Dispatch::ConsumeRTMPImage, this, cam_id);


        cout << "is add : " << isAdd << endl;
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


    lock = mConMutexRTMP[mode];
    queue = &mQueue_rtmp[mode];
    con_v_wait = mCon_rtmp[mode];
    rtmpLock = mRtmpMutex[mode];
    writer = mRTMPWriter[mode];

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

        img = queue->front();
        queue->pop();
        guard.unlock();

//        cout<< mode << " rtmp img queue size : " << queue->size() << endl;

        rtmpLock->lock();
//        TODO 推流逻辑
//        rtmpHandler->pushRTMP(img);
        writer.write(img);
        rtmpLock->unlock();
        num++;
        if (num == 10000) num = 0;
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
//    cv::VideoWriter writer;

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
//    writer = mRTMPWriter[mode];


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

    cout << "mode "<< mode << "camera total suc "<< getCurrentTime() << endl;

    int num = 0;
    int64_t end = getCurrentTime();

    int circle_i = 0;
    while (mCamLive[mode]) {
        circle_i ++;
        cam.read(frame);
//        cout << "ProduceImage "<< mode << " img "<< circle_i << " cost : " << getCurrentTime()-end << endl;
        end = getCurrentTime();

        if (circle_i % frames_skip != 0 ){
            if (circle_i==10000) circle_i = 0;
            if (rtmp_mode == 1){
                rtmpLock->lock();
//                rtmpHandler->pushRTMP(*rtmp_img);
//                writer.write(*rtmp_img);
                rtmpLock->unlock();
            }
            continue;
        }

        num ++;
        if (num == 10000) num = 0;
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

        if (circle_i==10000) circle_i = 0;
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
    SSD_Detection ssd_detection = SSD_Detection();
//    cv::VideoWriter writer;

    lock = mConMutexCam[mode];
//    lock = &myMutex_front;
    queue = &mQueueCam[mode];
    rtmpQueue = &mQueue_rtmp[mode];
    con_v_wait = mCon_not_empty[mode];
    con_v_notification = mCon_not_full[mode];
//            rtmpHandler = &ls_handler_front;
    con_rtmp = mCon_rtmp[mode];
    rtmp_img = &mRtmpImg[mode];
//    writer = mRTMPWriter[mode];

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
//        ret_img = frame;

        std::vector<int> hf_boxs;
        std::vector<std::vector<int>> ldmk_boxes;
        int64_t start = getCurrentTime();
        ssd_detection.detect_hf(ret_img, hf_boxs);
        cout <<"mode : " << mode << " ssd cost : " << getCurrentTime()-start << endl;
        for (int i = 0; i < hf_boxs.size(); i+=6) {
            if (hf_boxs[i+5]==2){
                std::vector<int> box_tmp = {hf_boxs[i],hf_boxs[i+1],hf_boxs[i+2],hf_boxs[i+3]};
                std::cout << hf_boxs[i] << " " <<hf_boxs[i+1]<<" " <<hf_boxs[i+2]<<" " <<hf_boxs[i+3]<<std::endl;
                if (ldmk_boxes.size() < 8) ldmk_boxes.emplace_back(box_tmp);
            }
            cv::Point p1, p2;
            p1.x = hf_boxs[i];
            p1.y = hf_boxs[i+1];
            p2.x = hf_boxs[i+2];
            p2.y = hf_boxs[i+3];
            cv::Scalar color = cv::Scalar(0, 255, 255);
            if (hf_boxs[i+5]==2) color = cv::Scalar(0, 255, 0);
            cv::rectangle(ret_img, p1, p2, color, 2, 1, 0);
        }

        if (ldmk_boxes.size()>0){
            std::vector<std::vector<int>>rects;
            std::vector<std::vector<float>>angles;
            int64_t start_kpt = getCurrentTime();
            ssd_detection.get_angles(ret_img,ldmk_boxes,angles);
            int64_t start_age = getCurrentTime();
            ssd_detection.get_ageGender(ret_img,ldmk_boxes,angles);
            cout <<"mode : " << mode << " kpt cost : " << start_age-start_kpt << endl;
            cout <<"mode : " << mode << " age cost : " << getCurrentTime()-start_age << endl;
        }
        cout <<"mode : " << mode << " total cost : " << getCurrentTime()-start << endl;

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
//            cv::Mat frame_clone = rtmp_frame;
            *rtmp_img = frame_clone;
//            int64_t start = getCurrentTime();

            rtmpQueue->push(*rtmp_img);
//            writer.write(*rtmp_img);
//            int64_t end = getCurrentTime();
//            cout << "writer rtmp cost : " << end - start << endl;
            con_rtmp->notify_all();
        }
//        cout << "Consumer number : " << num << endl;
        num++;
        if (num == 10000) num = 0;
    }

}

void Dispatch::multithreadTest(){

    vector<thread> threadArr;
    for (int i = 0; i < mCamLive.size(); ++i) {
        if (i==0){
            threadArr.emplace_back(&Dispatch::RPCServer, this);
        }
        if (mCamLive[i]){
            cout << "start woker thread " << endl;
            threadArr.emplace_back(&Dispatch::ProduceImage, this, i);
            threadArr.emplace_back(&Dispatch::ConsumeImage, this, i);
            threadArr.emplace_back(&Dispatch::ConsumeRTMPImage, this, i);
        }
    }

    for (auto& t : threadArr) {
        t.join();
    }

}
