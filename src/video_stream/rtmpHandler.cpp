//
// Created by Orangels on 2019-10-24.
//

#include "rtmpHandler.h"

rtmpHandler::rtmpHandler(){

}

//rtmpHandler::rtmpHandler(){
//    inUrl = "rtsp://192.168.88.29:554/user=admin&password=admin&channel=1&stream=0.sdp?real_stream";
//    //nginx-rtmp 直播服务器rtmp推流URL
//    outUrl = "rtmp://192.168.88.27:1935/hls/test_generate";
//
//    avcodec_register_all();
//
//    av_register_all();
//
//    avformat_network_init();
//
//    vsc = NULL;
//
//    yuv = NULL;
//
//    vc = NULL;
//
//    ic = NULL;
//
//    inWidth = 1920;
//    inHeight = 1080;
//    fps = 15;
//
//    vsc = sws_getCachedContext(vsc,
//                               inWidth, inHeight, AV_PIX_FMT_BGR24,     //源宽、高、像素格式
//                               inWidth, inHeight, AV_PIX_FMT_YUV420P,//目标宽、高、像素格式
//                               SWS_BICUBIC,  // 尺寸变化使用算法
//                               0, 0, 0
//    );
//    yuv = av_frame_alloc();
//    yuv->format = AV_PIX_FMT_YUV420P;
//    yuv->width = inWidth;
//    yuv->height = inHeight;
//    yuv->pts = 0;
//    //分配yuv空间
//    int ret = av_frame_get_buffer(yuv, 32);
//    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
//    vc = avcodec_alloc_context3(codec);
//    //c 配置编码器参数
//    vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; //全局参数
//    vc->codec_id = codec->id;
//    vc->thread_count = 8;
//
//    vc->bit_rate = 50 * 1024 * 8;//压缩后每秒视频的bit位大小 50kB
//    vc->width = inWidth;
//    vc->height = inHeight;
//    vc->time_base = { 1,fps };
//    vc->framerate = { fps,1 };
//
//    //画面组的大小，多少帧一个关键帧
//    vc->gop_size = 50;
//    vc->max_b_frames = 0;
//    vc->pix_fmt = AV_PIX_FMT_YUV420P;
//    //d 打开编码器上下文
//    ret = avcodec_open2(vc, 0, 0);
//    //a 创建输出封装器上下文
//    ret = avformat_alloc_output_context2(&ic, 0, "flv", outUrl);
//    //b 添加视频流
//    AVStream *vs = avformat_new_stream(ic, NULL);
//    vs->codecpar->codec_tag = 0;
//    //从编码器复制参数
//    avcodec_parameters_from_context(vs->codecpar, vc);
//    av_dump_format(ic, 0, outUrl, 1);
//
//
//    ///打开rtmp 的网络输出IO
//    ret = avio_open(&ic->pb, outUrl, AVIO_FLAG_WRITE);
//    //写入封装头
//    ret = avformat_write_header(ic, NULL);
//    memset(&pack, 0, sizeof(pack));
//}

rtmpHandler::rtmpHandler(string inUrl_parm, string outUrl_parm, int width, int height, int fps_parm){
    std::cout << "111" << std::endl;
    avcodec_register_all();
    av_register_all();
    avformat_network_init();

    vsc = NULL;
    yuv = NULL;
    vc = NULL;
    ic = NULL;

    inWidth = width;
    inHeight = height;
    fps = fps_parm;
    outUrl =  outUrl_parm.c_str();

    vsc = sws_getCachedContext(vsc,
                               inWidth, inHeight, AV_PIX_FMT_BGR24,     //源宽、高、像素格式
                               inWidth, inHeight, AV_PIX_FMT_YUV420P,//目标宽、高、像素格式
                               SWS_BICUBIC,  // 尺寸变化使用算法
                               0, 0, 0
    );
    yuv = av_frame_alloc();
    yuv->format = AV_PIX_FMT_YUV420P;
    yuv->width = inWidth;
    yuv->height = inHeight;
    yuv->pts = 0;
    // 分配yuv空间
    int ret = av_frame_get_buffer(yuv, 32);
    //a 找到编码器
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    //b 创建编码器上下文
    vc = avcodec_alloc_context3(codec);
    //c 配置编码器参数
    vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; //全局参数
    vc->codec_id = codec->id;
    vc->thread_count = 8;
//    vc->thread_count = 4;
//    vc->thread_count = 18;

    vc->bit_rate = 30 * 1024 * 8;//压缩后每秒视频的bit位大小 50kB
//    vc->bit_rate = 70 * 1024 * 8;//压缩后每秒视频的bit位大小 50kB
//    vc->bit_rate = 50 * 1024 * 8;//压缩后每秒视频的bit位大小 50kB
//    vc->rc_max_rate = 50 * 1024 * 8;
//    vc->rc_min_rate = 50 * 1024 * 8;
    vc->width = inWidth;
    vc->height = inHeight;
    vc->time_base = { 1,fps };
    vc->framerate = { fps,1 };

    //画面组的大小，多少帧一个关键帧
//    vc->gop_size = 50;
    vc->gop_size = 30;
    vc->max_b_frames = 0;
    vc->pix_fmt = AV_PIX_FMT_YUV420P;

    //延时编码新增
//    av_opt_set(vc->priv_data, "preset", "superfast", 0);

    // 实时编码关键看这句，上面那条无所谓
    av_opt_set(vc->priv_data, "tune", "zerolatency", 0);

    //d 打开编码器上下文
    ret = avcodec_open2(vc, 0, 0);
    //a 创建输出封装器上下文
    ret = avformat_alloc_output_context2(&ic, 0, "flv", outUrl);
    //b 添加视频流
    vs = avformat_new_stream(ic, NULL);
    vs->codecpar->codec_tag = 0;
    //从编码器复制参数
    avcodec_parameters_from_context(vs->codecpar, vc);
    av_dump_format(ic, 0, outUrl, 1);


    ///打开rtmp 的网络输出IO
    ret = avio_open(&ic->pb, outUrl, AVIO_FLAG_WRITE);
    //写入封装头
    ret = avformat_write_header(ic, NULL);
    memset(&pack, 0, sizeof(pack));
}

rtmpHandler::rtmpHandler(const char * inUrl_parm, const char * outUrl_parm, int width, int height, int fps_parm){
    std::cout << "222" << std::endl;
    avcodec_register_all();
    av_register_all();
    avformat_network_init();

    vsc = NULL;
    yuv = NULL;
    vc = NULL;
    ic = NULL;

    inWidth = width;
    inHeight = height;
    fps = fps_parm;
    outUrl =  outUrl_parm;

    vsc = sws_getCachedContext(vsc,
                               inWidth, inHeight, AV_PIX_FMT_BGR24,     //源宽、高、像素格式
                               inWidth, inHeight, AV_PIX_FMT_YUV420P,//目标宽、高、像素格式
                               SWS_BICUBIC,  // 尺寸变化使用算法
                               0, 0, 0
    );
    yuv = av_frame_alloc();
    yuv->format = AV_PIX_FMT_YUV420P;
    yuv->width = inWidth;
    yuv->height = inHeight;
    yuv->pts = 0;
    // 分配yuv空间
    int ret = av_frame_get_buffer(yuv, 32);
    //a 找到编码器
    codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    //b 创建编码器上下文
    vc = avcodec_alloc_context3(codec);
    //c 配置编码器参数
    vc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER; //全局参数
    vc->codec_id = codec->id;
//    vc->thread_count = 8;
    vc->thread_count = 6;
//    vc->thread_count = 12;

//    vc->bit_rate = 200 * 1024 * 8;//压缩后每秒视频的bit位大小 50kB
    vc->bit_rate = 30 * 1024 * 8;//压缩后每秒视频的bit位大小 50kB
//    vc->bit_rate = 50 * 1024 * 8;//压缩后每秒视频的bit位大小 50kB
//    vc->rc_max_rate = 50 * 1024 * 8;
//    vc->rc_min_rate = 50 * 1024 * 8;
    vc->width = inWidth;
    vc->height = inHeight;
    vc->time_base = { 1,fps };
    vc->framerate = { fps,1 };

    //画面组的大小，多少帧一个关键帧
//    vc->gop_size = 10;
    vc->gop_size = 30;
    vc->max_b_frames = 0;
    vc->pix_fmt = AV_PIX_FMT_YUV420P;

    //延时编码新增
    av_opt_set(vc->priv_data, "preset", "superfast", 0);

    // 实时编码关键看这句，上面那条无所谓
    av_opt_set(vc->priv_data, "tune", "zerolatency", 0);

    //d 打开编码器上下文
    ret = avcodec_open2(vc, 0, 0);
    //a 创建输出封装器上下文
    ret = avformat_alloc_output_context2(&ic, 0, "flv", outUrl);
    //b 添加视频流
    vs = avformat_new_stream(ic, NULL);
    vs->codecpar->codec_tag = 0;
    //从编码器复制参数
    avcodec_parameters_from_context(vs->codecpar, vc);
    av_dump_format(ic, 0, outUrl, 1);


    ///打开rtmp 的网络输出IO
    ret = avio_open(&ic->pb, outUrl, AVIO_FLAG_WRITE);
    //写入封装头
    ret = avformat_write_header(ic, NULL);
    memset(&pack, 0, sizeof(pack));
}

void rtmpHandler::pushRTMP(cv::Mat frame){
    int ret = 0;
//    for (;;)
//    {
//        ///读取rtsp视频帧，解码视频帧
//        if (!cam.grab())
//        {
//            continue;
//        }
//        ///yuv转换为rgb
//        if (!cam.retrieve(frame))
//        {
//            continue;
//        }
////            imshow("video", frame);
////            waitKey(20);


        ///rgb to yuv
        //输入的数据结构
        uint8_t *indata[AV_NUM_DATA_POINTERS] = { 0 };
        //indata[0] bgrbgrbgr
        //plane indata[0] bbbbb indata[1]ggggg indata[2]rrrrr
        indata[0] = frame.data;
        int insize[AV_NUM_DATA_POINTERS] = { 0 };
        //一行（宽）数据的字节数
        insize[0] = frame.cols * frame.elemSize();
        int h = sws_scale(vsc, indata, insize, 0, frame.rows, //源数据
                          yuv->data, yuv->linesize);
        if (h <= 0)
        {
            return;
        }
        //cout << h << " " << flush;
        ///h264编码
        yuv->pts = vpts;
        vpts++;
        ret = avcodec_send_frame(vc, yuv);
        if (ret != 0)
            return;

        ret = avcodec_receive_packet(vc, &pack);
        if (ret != 0 || pack.size > 0)
        {
//            cout << "*" << pack.size << flush;
        }
        else
        {
            return;
        }
        //推流
        pack.pts = av_rescale_q(pack.pts, vc->time_base, vs->time_base);
        pack.dts = av_rescale_q(pack.dts, vc->time_base, vs->time_base);
        pack.duration = av_rescale_q(pack.duration, vc->time_base, vs->time_base);
        ret = av_interleaved_write_frame(ic, &pack);
        if (ret == 0)
        {
//            cout << "#" << flush;
        }
//    }
}