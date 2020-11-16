//
// Created by Orangels on 2020-04-19.
//

#include "dsGSTHandler.h"


int64_t getCurrentTime_ds()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}



dsHandler::dsHandler(){

}

dsHandler::dsHandler(string vRTSPCAM, int vMUXER_OUTPUT_WIDTH, int vMUXER_OUTPUT_HEIGHT, int vMUXER_BATCH_TIMEOUT_USEC,
                     int camNum, int mode, int vFrame_skip) :
        RTSPCAM(vRTSPCAM), MUXER_OUTPUT_WIDTH(vMUXER_OUTPUT_WIDTH), MUXER_OUTPUT_HEIGHT(vMUXER_OUTPUT_HEIGHT),
        MUXER_BATCH_TIMEOUT_USEC(vMUXER_BATCH_TIMEOUT_USEC), frame_skip(vFrame_skip){

    //    mode 0 h264, mode 1 h265
    string camPath = "";

    switch (mode) {
        case 0:
            camPath = "rtspsrc location=" + vRTSPCAM +
                      " latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)" +
                      to_string(vMUXER_OUTPUT_WIDTH) + ", height=(int)" + to_string(vMUXER_OUTPUT_HEIGHT) +
                      ", format=(string)BGRx ! videoconvert ! appsink";
            break;
        case 1:
            camPath = "rtspsrc location=" + vRTSPCAM +
                      " latency=0 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)" +
                      to_string(vMUXER_OUTPUT_WIDTH) + ", height=(int)" + to_string(vMUXER_OUTPUT_HEIGHT) +
                      ", format=(string)BGRx ! videoconvert ! appsink";
            break;
    }
    RTSPCAM        = camPath;
}

void dsHandler::run(){
    int num = 0;
    state = 1;
    cv::Mat frame;

    cam.open(RTSPCAM);

    if (!cam.isOpened()) {
        cout << "cam open failed!" << endl;
        return;
    }

    int64_t start = getCurrentTime_ds();

    while (state != 0) {
        cam.read(frame);
        num++;

        if (num == 100000) num = 0;
        if (num % frame_skip == 0) {

            std::unique_lock <std::mutex> guard(myMutex);
            while (imgQueue.size() >= 20) {
                std::cout << "Produce  is waiting for items...\n";
                mCon_not_full.wait(guard);
            }

            cout << "rtsp " << num << " cost -- " << getCurrentTime_ds() - start << endl;
            imgQueue.push(frame);
            mCon_not_empty.notify_all();
            guard.unlock();

            start = getCurrentTime_ds();

            //            if (imgQueue.size() < 10) {
            //                imgQueue.push(frame);
            //                con_v_notification.notify_all();
            //            }
            //            guard.unlock();

        }

    }
}

void dsHandler::finish(){
    state = 0;
}