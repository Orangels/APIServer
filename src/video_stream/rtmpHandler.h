//
// Created by Orangels on 2019-10-24.
//

#ifndef FFMPEG_RTMP_RTMPHANDLER_H
#define FFMPEG_RTMP_RTMPHANDLER_H
#include <opencv2/highgui.hpp>
#include <iostream>

extern "C"
{
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
}

using namespace std;

class rtmpHandler {
    public :
        const char * inUrl;
        const char * outUrl;
        int inWidth;
        int inHeight;
        int fps;


        rtmpHandler();
        rtmpHandler(const char * inUrl_parm, const char * outUrl_parm, int width, int height, int fps_parm);
        rtmpHandler(string inUrl_parm, string outUrl_parm, int width, int height, int fps_parm);
        void pushRTMP(cv::Mat frame);

    private:
        //像素格式转换上下文
        SwsContext *vsc = NULL;

        //输出的数据结构
        AVFrame *yuv = NULL;

        //编码器上下文
        AVCodecContext *vc = NULL;

        //rtmp flv 封装器
        AVFormatContext *ic = NULL;

        //编码器
        AVCodec *codec;
        //视频流
        AVStream *vs;
        AVPacket pack;
        int vpts = 0;

};


#endif //FFMPEG_RTMP_RTMPHANDLER_H
