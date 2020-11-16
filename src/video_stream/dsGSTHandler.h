//
// Created by Orangels on 2020-04-19.
//

#ifndef GSTREAM_DEMO_dsHandler_H
#define GSTREAM_DEMO_dsHandler_H
#include "stdio.h"
#include "gst/gst.h"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include "gstnvdsmeta.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include <iostream>

#include <functional>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>


using namespace std;
using namespace cv;

Mat writeImage(GstBuffer *buf);

class dsHandler {
public:
    int MUXER_OUTPUT_WIDTH;
    int MUXER_OUTPUT_HEIGHT;
    int MUXER_BATCH_TIMEOUT_USEC;
    string RTSPCAM;

    dsHandler();
    dsHandler(string vRTSPCAM, int vMUXER_OUTPUT_WIDTH, int vMUXER_OUTPUT_HEIGHT, int vMUXER_BATCH_TIMEOUT_USEC, int camNum, int mode, int vFrame_skip);

    ~dsHandler(){
        cout << "dsHandler del" << endl;
    };
    void run();
    void finish();

    queue<cv::Mat> imgQueue;
    mutex myMutex;
    condition_variable mCon_not_empty, mCon_not_full;


    int frame_skip = 1;
    int pic_num = 0;
    cv::VideoCapture cam;
    bool state = 0;  // 0: 结束拉流 1:暂停拉流  2:拉流
private:
    GstElement *pipeline = NULL, *source = NULL, *rtppay = NULL, *parse = NULL,
               *decoder = NULL, *sink = NULL, *filter1 = NULL;
    GstCaps *filtercaps = NULL;
};


#endif //GSTREAM_DEMO_dsHandler_H
