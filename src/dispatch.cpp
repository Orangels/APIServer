//
// Created by xinxueshi on 2020/4/28.
//  https://blog.csdn.net/yiyouxian/article/details/51993524
//
#include <iostream>
#include <iomanip>
#include <Python.h>
#include <numpy/arrayobject.h>
#include "dispatch.h"
#include "config.h"
#include "Common.h"
#include "pythonCaller.h"
#include "utils/mat2numpy.h"
#include "utils/vis.h"
#include "utils/split.h"
#include "utils/track.h"
#include "utils/match_id.h"
#include <sys/time.h>
#include <gflags/gflags.h>
#include "fishCamHandler/cameraHandler.h"
#include <opencv2/imgproc/imgproc.hpp>
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"



using namespace std;

DEFINE_string(lens, "vivotek", "camera lens");
//DEFINE_string(mode, "center", "dewarping mode");
DEFINE_string(mode, "perimeter", "dewarping mode");
DEFINE_string(input, "rtsp://root:admin123@192.168.88.67/live.sdp", "input video file name");
//DEFINE_string(input, "rtsp://root:admin123@192.168.88.26/live.sdp", "input video file name");
DEFINE_string(output, "perimeter_ls_test.mp4", "output video file name");

DEFINE_double(camera_rotation, 0.0, "camera rotation degree");
DEFINE_double(pixels_per_degree, 16.0, "pixels per degree");
DEFINE_double(center_zoom_angle, 90.0, "center zoom field of view");
DEFINE_double(perimeter_top_angle, 90.0, "perimeter top angle");
DEFINE_double(perimeter_bottom_angle, 30.0, "perimeter bottom angle");


static Person_ID * Person_id = new Person_ID();

int64_t getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

cv::Size CalculateSize(string mode="center") {
    if (mode == "center") {
        int length = int(FLAGS_pixels_per_degree *
                         FLAGS_center_zoom_angle / 4) << 2;
        return cv::Size(length, length);
    } else {
        int width = int(FLAGS_pixels_per_degree * 180.0f / 4) << 2;
        int height = int(FLAGS_pixels_per_degree *
                         (FLAGS_perimeter_top_angle - FLAGS_perimeter_bottom_angle)
                         / 2) << 2;
        return cv::Size(width, height);
    }
}


static void *test_function(gearman_job_st *job,
                           void *context,
                           size_t *result_size,
                           gearman_return_t *ret_ptr)
{
    (void)context;

    const char *workload;
    workload= (const char *)gearman_job_workload(job);
    *result_size= gearman_job_workload_size(job);

    uint64_t count= 0;
//    if (workload != NULL)
//    {
//        if (workload[0] != ' ' && workload[0] != '\t' && workload[0] != '\n')
//            count++;
//
//        for (size_t x= 0; x < *result_size; x++)
//        {
//            if (workload[x] != ' ' && workload[x] != '\t' && workload[x] != '\n')
//                continue;
//
//            count++;
//
//            while (workload[x] == ' ' || workload[x] == '\t' || workload[x] == '\n')
//            {
//                x++;
//                if (x == *result_size)
//                {
//                    count--;
//                    break;
//                }
//            }
//        }
//    }


    std::string result= "asdasd";
    std::cout << *result_size << endl;



    cout << "mode 2 json" << endl;
    string jsonStr = workload;
    jsonStr= jsonStr.substr(0, static_cast<int>(*result_size));
    cout << "params json : " << jsonStr << endl;
    rapidjson::Document dom;
    if (!dom.Parse(jsonStr.c_str()).HasParseError()) {
        if (dom.HasMember("persons") && dom["persons"].IsArray()){
            const rapidjson::Value& arr = dom["persons"];

            for (int i = 0; i < arr.Size(); ++i) {
                if (arr[i].HasMember("id") && arr[i]["id"].IsInt()) {
                    int person_id = arr[i]["id"].GetInt();
                    cout << "person id : " << person_id << endl;
                    Person_id->person_id.push_back(person_id);
                }
                if (arr[i].HasMember("timestamp") && arr[i]["timestamp"].IsInt64()) {
                    auto timestamp = arr[i]["timestamp"].GetInt64();
                    cout << "timestamp is : " << timestamp << endl;
                    Person_id->timestamp.push_back(timestamp);
                }
                if (arr[i].HasMember("img_path") && arr[i]["img_path"].IsString()) {
                    auto img_str = arr[i]["img_path"].GetString();
                    cout << "img_path is : " << img_str << endl;
//                    Person_id.paths.push_back(img_str);
                }
            }
        }
    }else{
        printf("fail to parse json str\n");
    }



    *result_size= result.size();

    *ret_ptr= GEARMAN_SUCCESS;
    return strdup(result.c_str());
}


Dispatch::Dispatch()
{
    //    ls add
//    camera_mid = false;
//    camera_front = false;
    camera_mid = true;
    camera_front = true;
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
        ls_handler_front = rtmpHandler("",rtmpPath_0,out_w,out_h,fps);
//        ls_handler_mid = rtmpHandler("",rtmpPath_1,out_w,out_h,fps);
    }

    char* gearSvrHost=(char*)"127.0.0.1", *gearSvrPort=(char*)"4730";
    char* gearContext = new char(50);

    gearWorker = gearman_worker_create(NULL);
    if (gearWorker == NULL)
    {
        cout << "ERROR: " << gearman_worker_error(gearWorker) << endl;
    }

    gearWRet = gearman_worker_add_server(gearWorker, gearSvrHost, atoi(gearSvrPort));
    if (gearman_failed(gearWRet))
    {
        cout << "ERROR: " << gearman_worker_error(gearWorker) << endl;
    }


    int tracker_mode = stoi(labels["TRACK_MODE"]);
    if (stoi(labels["inference_switch"])){
        int ret = 0;
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
        if (tracker_mode == 0){
            pyEngineAPI_0 =  new Engine_api("engine_api");
            pyEngineAPI_1 =  new Engine_api("engine_api");
            pyEngineAPI_2 =  new Engine_api("engine_api");
        } else{
            cout << "init py start" << endl;
            pyEngineAPI_0 =  new Engine_api("tracker_api");
            pyEngineAPI_1 =  new Engine_api("tracker_api");
            pyEngineAPI_2 =  new Engine_api("tracker_api");
            cout << "init py end" << endl;
        }

        matcher = new Match_ID(stoi(labels["HEAD_TRACK_MISTIMES"]), stoi(labels["IMAGE_W"]), stoi(labels["IMAGE_H"]), stoi(labels["DIS"]));
    }

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
    char* gearContext = new char(50);
    gearWRet = gearman_worker_add_function(gearWorker,
                                          "DPH_TRACKER_SERVER",
                                          0,
                                          test_function,
                                          gearContext);



    if (gearman_failed(gearWRet))
    {
        cout << "ERROR: " << gearman_worker_error(gearWorker) << endl;
    }

    cout << "waiting for job ... " << endl;

    while (1)
    {
        gearWRet = gearman_worker_work(gearWorker);
        if (gearman_failed(gearWRet))
        {
            cout << "ERROR: " << gearman_worker_error(gearWorker) << endl;
            break;
        }
    }

    delete gearContext;
    gearman_worker_free(gearWorker);
}

void Dispatch::ConsumeRTMPImage(int mode){

    cv::Mat img;
    int num = 0;

    mutex *lock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait;
    mutex* rtmpLock;
    rtmpHandler* rtmpHandler;

    switch (mode) {
        case 0:
            lock = &myMutex_rtmp_front;
            queue = &mQueue_rtmp_front;
            con_v_wait = &con_rtmp_front;
            rtmpLock = &rtmpMutex_front;
            rtmpHandler = &ls_handler_front;
            break;
        case 1:
            lock = &myMutex_rtmp_mid;
            queue = &mQueue_rtmp_mid;
            con_v_wait = &con_rtmp_mid;
            rtmpLock = &rtmpMutex_mid;
            rtmpHandler = &ls_handler_mid;
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
        rtmpHandler->pushRTMP(img);
        rtmpLock->unlock();
        num++;
    }
}

void Dispatch::ProduceImage(int mode){
//    dsHandler dsHandler_ls("rtsp://admin:sx123456@192.168.88.38:554/h264/ch2/main/av_stream",
//                           1280,720,4000000);
//    dsHandler dsHandler_ls("rtsp://192.168.88.29:554/user=admin&password=admin&channel=1&stream=0.sdp?real_stream",
//                           1920,1080,4000000, 1);
//    dsHandler_ls.run();

//    Cconfig labels = Cconfig("../cfg/process.ini");
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
    queue<vector<cv::Mat>> *queue;
    condition_variable *con_v_wait, *con_v_notification;
    rtmpHandler* rtmpHandler;
    cv::Mat *rtmp_img;

    switch (mode){
        case 0:
            path = path_0;
            lock = &myMutex_front;
            queue = &mQueue_front;
            con_v_wait = &con_front_not_full;
            con_v_notification = &con_front_not_empty;
            rtmpHandler = &ls_handler_front;
            rtmpLock = &rtmpMutex_front;
            rtmp_img = &rtmp_front_img;
            break;
        case 1:
            path = path_1;
            lock = &myMutex_mid;
            queue = &mQueue_mid;
            con_v_wait = &con_mid_not_full;
            con_v_notification = &con_mid_not_empty;
            rtmpHandler = &ls_handler_mid;
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

    if (mode == 0){
        camera_front = cam.isOpened();
    } else if (mode == 1){
        camera_mid = cam.isOpened();
    }

    while (camera_mid != true || camera_front != true){
        continue;
    }

    if (!cam.isOpened())
    {
        cout << "cam open failed!" << endl;
        return;
    }

    cout << "mode "<< mode << "camera total suc "<< getCurrentTime() << endl;

    cv::Size size_center = CalculateSize("center");
    cv::Size size_perimeter = CalculateSize("perimeter");
    cv::Mat dst_center(size_center, CV_8UC3);
    cv::Mat dst_perimeter(size_perimeter, CV_8UC3);

    cameraHandler* camera_center = new cameraHandler(frame.cols, frame.rows, frame.data,
                                                     dst_center.cols, dst_center.rows, dst_center.data,
                                                     FLAGS_center_zoom_angle);

    cameraGHandler* camera_perimeter = new cameraGHandler(frame.cols, frame.rows, frame.data,
                                                          dst_perimeter.cols, dst_perimeter.rows, dst_perimeter.data,
                                                          FLAGS_perimeter_top_angle,
                                                          FLAGS_perimeter_bottom_angle);

    int sum = 0;
    int num = 0;
    int64_t end = getCurrentTime();
    for (int i=0; ; i++) {
        //        TODO 跳帧
        cam.read(frame);

//        cout << "ProduceImage "<< mode << " img "<< i << " cost : " << getCurrentTime()-end << endl;
        end = getCurrentTime();

        if (i % frames_skip != 0 ){
            if (rtmp_mode == 1){
                rtmpLock->lock();
                rtmpHandler->pushRTMP(*rtmp_img);
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


        int64_t start_fish = getCurrentTime();
        auto center_reader = [&](){
            camera_center->run(frame.data);
        };
        thread thread_center(center_reader);
        camera_perimeter->run(frame.data);
        thread_center.join();
        int64_t end_fish = getCurrentTime();

        sum += (end_fish-start_fish);
//        cout << "frame fish : " << num << " cost " << end_fish-start_fish << endl;
//        cout << "frame fish : " << num << " avg cost " << sum/num << endl;

        if (stoi(labels["SAVE_IAMGE"]))
        {
            string img_path, img_path_center, img_path_ori;
            img_path = "./imgs/" + to_string(num+10000) + ".jpg";
            img_path_center = "./imgs_center/" + to_string(num+10000) + ".jpg";
            img_path_ori = "./imgs_ori/" + to_string(num+10000) + ".jpg";
            cv::imwrite(img_path, dst_perimeter);
            cv::imwrite(img_path_center, dst_center);
        }

        vector<cv::Mat> queue_item = { frame, dst_perimeter, dst_center };
        queue->push(queue_item);
        con_v_notification->notify_all();
        guard.unlock();
//        int64_t end = getCurrentTime();
//        cout << "write every time  : " << mode << "  " << i << " -- " << (end - start)  << endl;
//        cout << "write process time  : " << mode << "  " << num << " -- " << (end - start)  << endl;
    }

}

int init_numpy()
{
    import_array();
}

void Dispatch::ConsumeImage(int mode){
    int out_w = stoi(labels["OUT_W"]);  //2880
    int out_h = stoi(labels["OUT_H"]);  //960

    int num = 0;
    vector<cv::Mat> queue_item;
    cv::Mat frame;
    cv::Mat dst_perimeter;
    cv::Mat dst_center;

    mutex *lock;
    queue<cv::Mat> *rtmpQueue;
    queue<vector<cv::Mat>> *queue;
    condition_variable *con_rtmp;
    condition_variable *con_v_wait, *con_v_notification;
    cv::Mat* rtmp_img;

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
    cv::Mat ret_img;


    while (true){
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->empty()) {
//            std::cout << "Consumer " << mode <<" is waiting for items...\n";
            con_v_wait->wait(guard);
        }

        queue_item = queue->front();
        queue->pop();
        con_v_notification->notify_all();
        guard.unlock();

        frame = queue_item[0];
        dst_perimeter = queue_item[1];
        dst_center = queue_item[2];

        //        TODO 业务逻辑
        int64_t start_fish = getCurrentTime();
        std::vector<cv::Mat> img_list;
        split(dst_perimeter,img_list,0);
//        cv::Mat tmp_test = cv::imread("/home/user/workspace/xxs/DPH_Server/data/test.png");
        cv::Mat result_img = dst_perimeter;
//        result_img.create(1920, 2880, tmp_test.type());

        if (stoi(labels["inference_switch"])){

            cv::Mat img0,img1,img2,img3;
//            img0 = img_list[0];
//            img1 = img_list[1];
//            img2 = img_list[2];
//            img3 = dst_center;
            cv::resize(img_list[0], img0, cv::Size(960, 480), 0, 0, cv::INTER_LINEAR);
            cv::resize(img_list[1], img1, cv::Size(960, 480), 0, 0, cv::INTER_LINEAR);
            cv::resize(img_list[2], img2, cv::Size(960, 480), 0, 0, cv::INTER_LINEAR);
            cv::resize(dst_center, img3, cv::Size(480, 480), 0, 0, cv::INTER_LINEAR);

            vector<int> vret0,vret1,vret2,vret3;

            int64_t end_resize = getCurrentTime();
            

            auto thread_func_0 = [&](){
                vret0 = pyEngineAPI_0->get_result(img0.clone(), "");
//                    LOG_DEBUG("pPer 0 \n");
            };
            auto thread_func_1 = [&](){
                vret1 = pyEngineAPI_1->get_result(img1.clone(), "");
//                    LOG_DEBUG("pPer 1 \n");
            };
            auto thread_func_2 = [&](){
                vret2 = pyEngineAPI_2->get_result(img2.clone(), "");
//                    LOG_DEBUG("pPer 2 \n");
            };
            thread thread_ctx_0(thread_func_0);
            thread thread_ctx_1(thread_func_1);
            thread thread_ctx_2(thread_func_2);

            thread_ctx_0.join();
            thread_ctx_1.join();
            thread_ctx_2.join();



////            result_img.create(1920, 2880, img0.type());
            cv::Mat tmp;
            tmp.create(480, 2880, dst_perimeter.type());
            cv::Mat r00 = tmp(cv::Rect(1920, 0, 960, 480));
            img0.copyTo(r00);
            cv::Mat r11 = tmp(cv::Rect(0, 0, 960, 480));
            img1.copyTo(r11);
            cv::Mat r22 = tmp(cv::Rect(960, 0, 960, 480));
            img2.copyTo(r22);


            matcher->MergeResult(tmp, result_img, num, vret0, vret1, vret2,stoi(labels["TRACK_MODE"]) == 0,Person_id);


            int64_t end_models = getCurrentTime();
            cv::Mat des;
            des.create(960, 1440, dst_perimeter.type());
            r00 = des(cv::Rect(0, 0, 1440, 480));
            img0 = result_img(cv::Rect(0, 0, 1440, 480));
            img0.copyTo(r00);
////
            r11 = des(cv::Rect(0, 480, 1440, 480));
            img1 = result_img(cv::Rect(1440, 0, 1440, 480));
            img1.copyTo(r11);

            ret_img.create(1920, 2880, dst_perimeter.type());

            cv::resize(des, ret_img, cv::Size(2880, 1920), 0, 0, cv::INTER_LINEAR);
            int64_t end_copy = getCurrentTime();
            if (stoi(labels["SAVE_DETECTIONS"])){
                string img_detections;
                img_detections = "./imgs_detections/" + to_string(num+10000) + ".jpg";
                cv::imwrite(img_detections, ret_img);
            }
            int64_t end_fish = getCurrentTime();
//            cout << "image resize : " << " cost " << end_resize-start_fish << endl;
//            cout << "model infer  : " << " cost " << end_models-end_resize << endl;
//            cout << "image copy   : " << " cost " << end_copy-end_models << endl;
//            cout << "image write  : " << " cost " << end_fish-end_copy << endl;
            cout << "process  all : " << " cost " << end_fish-start_fish << endl;
        }

        if (rtmp_mode == 1) {
            // 推流

            cv::Mat rtmp_frame;
//            cv::resize(dst_perimeter, rtmp_frame, cv::Size(out_w, out_h));
            if (stoi(labels["inference_switch"])){
                cv::resize(ret_img, rtmp_frame, cv::Size(out_w, out_h));
            } else{
                cv::resize(dst_perimeter, rtmp_frame, cv::Size(out_w, out_h));
            }
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
//    thread thread_write_image_mid(&Dispatch::ProduceImage, this, 1);

    thread thread_read_image_front(&Dispatch::ConsumeImage, this, 0);
//    thread thread_read_image_mid(&Dispatch::ConsumeImage, this, 1);

    thread thread_RTMP_front(&Dispatch::ConsumeRTMPImage, this, 0);
//    thread thread_RTMP_mid(&Dispatch::ConsumeRTMPImage, this, 1);

    thread thread_RPC_server(&Dispatch::RPCServer, this);

    thread_write_image_front.join();
//    thread_write_image_mid.join();

    thread_read_image_front.join();
//    thread_read_image_mid.join();

    thread_RTMP_front.join();
//    thread_RTMP_mid.join();

    thread_RPC_server.join();


}
// https://cloud.tencent.com/developer/ask/218359
int Dispatch::engine_Test(){
    Py_Initialize();//使用python之前，要调用Py_Initialize();这个函数进行初始化
    if (!Py_IsInitialized())
    {
        printf("初始化失败！");
        return 0;
    }
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../pycode')");//这一步很重要，修改Python路径
    PyObject * pModule = NULL;//声明变量
    PyObject * pFunc = NULL;// 声明变量
    PyObject * pClass = NULL;//声明变量
    PyObject * pInstance = NULL;
    PyObject* pName     = NULL;
    pName = PyUnicode_FromString("engine_api");
    if (pName == NULL) {
        PyErr_Print();
        throw std::invalid_argument("Error: PyUnicode_FromString");
    }
    pModule = PyImport_Import(pName);
    if (pModule == NULL) {
        PyErr_Print();
        throw std::invalid_argument("fails to import the module");
    }
//
    // 模块的字典列表
    PyObject* pDict = PyModule_GetDict(pModule);
    if (!pDict) {
        printf("Cant find dictionary./n");
        return -1;
    }

//     // 演示函数调用
//    cout<<"calling python function..."<<endl;
//    PyObject* pFunHi = PyDict_GetItemString(pDict, "hi_function");
//    PyObject_CallFunction(pFunHi, NULL, NULL);
//    Py_DECREF(pFunHi);


    cout<<"calling python class..."<<endl;
    // 演示构造一个Python对象，并调用Class的方法
    // 获取hi_class类
    PyObject* phi_class = PyDict_GetItemString(pDict, "ObjectApi");
    if (!phi_class ) {
        printf("Cant find phi_class class.\n");
        return -1;
    }
    PyObject* pInstance_hi_class = PyInstanceMethod_New(phi_class);

//    ls add 获取instance
    PyObject* pIns = PyObject_CallObject(pInstance_hi_class,nullptr);


    if (!pInstance_hi_class) {
        printf("Cant create instance.\n");
        return -1;
    }

    //调用hi_class类实例pInstance_hi_class里面的方法
//    ls change
//    PyObject_CallMethod(phi_class, "get_result", "O", pInstance_hi_class );
    PyObject_CallMethod(pIns,"get_result", nullptr);

    Py_DECREF(pIns);
    //释放
    Py_DECREF(phi_class);
    Py_DECREF(pInstance_hi_class );
    Py_DECREF(pModule);
    Py_Finalize(); // 与初始化对应
//
//    //获取calc类
//    PyObject* pClassCalc = PyDict_GetItemString(pDict, "ObjectApi");
//    if (!pClassCalc) {
//        printf("Cant find ObjectApi class.\n");
//        return -1;
//    }
//
//    //构造Python的实例
//    PyObject* pInstanceCalc = PyInstanceMethod_New(pClassCalc);
//    if (!pInstanceCalc) {
//        printf("Cant find calc instance.\n");
//        return -1;
//    }
//
//    PyObject* pRet = PyObject_CallMethod(pClassCalc,"show_result","", pInstanceCalc);
//    if (!pRet)
//    {
//        printf("不能找到 pRet");
//        return -1;
//    }
//
//    int res = 0;
//    PyArg_Parse(pRet, "i", &res);//转换返回类型
//
//    cout << "res:" << res << endl;//输出结果
//
//    Py_Finalize(); // 与初始化对应
////    system("pause");
    return 0;
}