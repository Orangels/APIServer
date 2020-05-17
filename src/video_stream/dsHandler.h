//
// Created by Orangels on 2020-04-19.
//

#ifndef GSTREAM_DEMO_DSHANDLER_H
#define GSTREAM_DEMO_DSHANDLER_H
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

#include <mutex>
#include <queue>
#include <condition_variable>


using namespace std;
using namespace cv;


class dsHandler {
    public:
        int MUXER_OUTPUT_WIDTH;
        int MUXER_OUTPUT_HEIGHT;
        int MUXER_BATCH_TIMEOUT_USEC;
        string RTSPCAM;

        dsHandler();
        dsHandler(string vRTSPCAM, int vMUXER_OUTPUT_WIDTH, int vMUXER_OUTPUT_HEIGHT, int vMUXER_BATCH_TIMEOUT_USEC, int mode);
        ~dsHandler(){

        };
        void run();

        static GstPadProbeReturn osd_sink_pad_buffer_probe (GstPad * pad, GstPadProbeInfo * info, gpointer u_data);
        static queue<cv::Mat> imgQueue;
        static mutex myMutex;
        static condition_variable con_v_notification;
        static int pic_num;
    private:
        GstElement *pipeline = NULL, *source = NULL, *rtppay = NULL, *parse = NULL,
            *decoder = NULL, *sink = NULL, *filter1 = NULL;
        GstCaps *filtercaps = NULL;
};


#endif //GSTREAM_DEMO_DSHANDLER_H
