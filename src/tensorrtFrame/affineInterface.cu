#include "affineInterface.h"


#define readPixel2(Pixel, X, Y) {Pixel.x = tex2D<uchar>(tex, X, Y); \
Pixel.y = tex2D<uchar>(tex, X+1.f, Y);\
Pixel.z = tex2D<uchar>(tex, X+2.f, Y);}

__device__ void appandColor2(float3 &voRBG, uchar3 vPixel, float xRelativity, float yRelativity) {
    float Relativity = xRelativity * yRelativity;
    voRBG.x += vPixel.x * Relativity;
    voRBG.y += vPixel.y * Relativity;
    voRBG.z += vPixel.z * Relativity;
}

__device__ void set_value(float3 *image, float3 data) {
    (*image).x = data.x;
    (*image).y = data.y;
    (*image).z = data.z;
}

__device__ void set_value_NCHW(float *image, int idx, int idy, int nWidth, int nHeight, float3 data) {
    int channel_size = nHeight * nWidth, moment = idy * nWidth + idx;
    image[moment + 0 * channel_size] = data.x;
    image[moment + 1 * channel_size] = data.y;
    image[moment + 2 * channel_size] = data.z;
}

__global__ void ImageRotate_kernel_tex(cudaTextureObject_t tex, float3 *poDataOut, int nWidth, int nHeight,
                                       int nNewWidth, int nNewHeight, float *inv_coeff, int num_boxes) {
    // now is BIP mode. Is BSQ necessary?
    int idy = blockIdx.y * blockDim.y + threadIdx.y;    //行
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    //列
//    int nWidth = srcData.m_srcWidth, nHeight = srcData.m_srcHeight;
//    __shared__ float inv_coeff_[8];/

    uchar3 TLPixel, TRPixel, BLPixel, BRPixel;
//    cudaTextureObject_t SrcImage = srcData.SrcImage;
//    __syncthreads();

    int memSize = nNewWidth * nNewHeight;

//    for (int i = 0; i < num_boxes; i++) {
    int i = idy / nNewHeight;
    int idx_new = idx % nNewWidth, idy_new = idy % nNewHeight;
//        memcpy(inv_coeff_, &inv_coeff[i], 4);

    float cosTheta = inv_coeff[i * 4 + 0], sinTheta = inv_coeff[i * 4 + 1],
            off_x = inv_coeff[i * 4 + 2], off_y = inv_coeff[i * 4 + 3];

    //get idx，idy  related coordination of source image
    float x2_map = cosTheta * idx_new + sinTheta * idy_new - off_x;
    float y2_map = -sinTheta * idx_new + cosTheta * idy_new - off_y;

    int x2 = (int) x2_map;
    int y2 = (int) y2_map;
    float xDistance = x2_map - x2;
    float yDistance = y2_map - y2;

    //idx_new < nNewWidth && idy_new < nNewHeight
    if (i < num_boxes) {
        //char3 to char, so how many step should the pointer moving
        int x_1byte = x2 * 3, y_1byte = y2;
        float3 RBG{0};

        if ((x2 >= 0) && (x2 + 1 < nWidth) && (y2 >= 0) && (y2 + 1 < nHeight)) {
            readPixel2(TLPixel, x_1byte, y_1byte);
            readPixel2(TRPixel, x_1byte + 3.0f, y_1byte);
            readPixel2(BLPixel, x_1byte, y_1byte + 1.0f);
            readPixel2(BRPixel, x_1byte + 3.0f, y_1byte + 1.0f);
            //printf("x:%d,y%d,z%d", TLPixel.x, TLPixel.y, TLPixel.z);
            appandColor2(RBG, TLPixel, 1.f - xDistance, 1.f - yDistance);
            appandColor2(RBG, TRPixel, xDistance, 1.f - yDistance);
            appandColor2(RBG, BLPixel, 1.f - xDistance, yDistance);
            appandColor2(RBG, BRPixel, xDistance, yDistance);
        } else if (((x2 + 1 == nWidth) && (y2 >= 0) && (y2 < nHeight)) ||
                   ((y2 + 1 == nHeight) && (x2 >= 0) && (x2 < nWidth))) {
            readPixel2(TLPixel, x_1byte, y_1byte);
            appandColor2(RBG, TLPixel, 1.f, 1.f);
        }
        //set step here if BSQ is necessary
//        int out_idx = idy_new * nNewWidth + idx_new + i * memSize;
        //set_value(&poDataOut[out_idx], RBG);//float3<>float3
        set_value_NCHW((float *) (poDataOut + i * memSize), idx_new, idy_new, nNewWidth, nNewHeight, RBG);
    }

}


__global__ void inv_transform_gpu(double *coeff, float *inv_coeff, int num_pts) {
    int i = threadIdx.x; // num_pts
    int num_paras = 6;//3*2

    if (i < num_pts) {
        float cosTheta = float(coeff[i * num_paras + 0]);
        float sinTheta = float(coeff[i * num_paras + 3]);
        float off_x = float(coeff[i * num_paras + 2]), _x;
        float off_y = float(coeff[i * num_paras + 5]), _y;
        float det = sinTheta * sinTheta + cosTheta * cosTheta;
        sinTheta /= det, cosTheta /= det;
        //求出idx，idy所在原始图像上的坐标
        _x = cosTheta * off_x + sinTheta * off_y;
        _y = -sinTheta * off_x + cosTheta * off_y;
//        std::cout << "inv coeff cossin : " << cosTheta << " "
//                  << sinTheta << " xy:" << _x << " " << _y << std::endl;
        inv_coeff[i * 4 + 0] = cosTheta;
        inv_coeff[i * 4 + 1] = sinTheta;
        inv_coeff[i * 4 + 2] = _x;
        inv_coeff[i * 4 + 3] = _y;
    }
}

__global__ void get_5kps_expand_gpu(float *kpts, float *points5, int batch, int pointsNum, int totalNum) {
    int key_point_pos[] = {36, 39, 42, 45, 33, 48, 54};// first 4 is eye_points
    int i = threadIdx.x;//10*batch
    int num_kpts = 10;//5*2
    int batch_i = i / num_kpts;
    int off = totalNum * 3 * batch_i;

    if (i < num_kpts * batch) {
        int j = i % num_kpts;//[0,10)
        if (j < 4) {
            int first = key_point_pos[j / 2 * 2], second = key_point_pos[j / 2 * 2 + 1];
            float x = (kpts[off + first + pointsNum * (j % 2)] + kpts[off + second + pointsNum * (j % 2)]) / 2;
            points5[num_kpts * batch_i + j] = x;
        } else {
            int first = key_point_pos[j / 2 + 2];
            float x = kpts[off + first + pointsNum * (j % 2)];
            points5[num_kpts * batch_i + j] = x;
        }
    }
}


Affine_::Affine_(int w, int h) : MAX_NUM_BOXES(25), out_w(w), out_h(h), show_eclipse_time(false) {
//    cudaStreamCreate(&stream);

    cudaEventCreate(&tic);
    cudaEventCreate(&toc);
    coeff_size = 4 * sizeof(float);
    exitIfCudaError(cudaMalloc(&dev_coeff, MAX_NUM_BOXES * coeff_size));
    exitIfCudaError(cudaMalloc(&dev_5kpts, MAX_NUM_BOXES * 10 * sizeof(float)));
    exitIfCudaError(cudaMalloc(&coeff_params, MAX_NUM_BOXES * 6 * sizeof(double)));
}

Affine_::~Affine_() {
    cudaFree(dev_coeff);
    cudaFree(dev_5kpts);
    cudaFree(coeff_params);
    //about event & stream
//    cudaStreamDestroy(stream);
    cudaEventDestroy(tic);
    cudaEventDestroy(toc);
    //reset*
    cudaDeviceReset();
}


void Affine_::re_malloc_coeff(int N) {
    if (N > MAX_NUM_BOXES) {
        coeff_size = 4 * sizeof(float);
        MAX_NUM_BOXES = (N + 9) / 10 * 10;
        cudaFree(dev_coeff);
        cudaFree(dev_5kpts);
        exitIfCudaError(cudaMalloc(&dev_coeff, coeff_size * MAX_NUM_BOXES));
        exitIfCudaError(cudaMalloc(&dev_5kpts, MAX_NUM_BOXES * 10 * sizeof(float)));
        exitIfCudaError(cudaMalloc(&coeff_params, MAX_NUM_BOXES * 6 * sizeof(double)));
    }
}

void Affine_::get_5_points_general(float *kpts, std::vector<float> &points5,
                                   int num_kpts, int pointsNum, int totalNum) {
    std::vector<int> key_point_pos{33, 48, 54}, eye_points{36, 39, 42, 45};
    int cur = 0;
    for (int i = 0; i < num_kpts; i++) {
        cur = totalNum * 3 * i;
        for (unsigned int j = 0; j < eye_points.size(); j += 2) {
            int first = eye_points[j], second = eye_points[j + 1];
            float x = (kpts[cur + first] + kpts[cur + second]) / 2;
            float y = (kpts[cur + first + pointsNum] + kpts[cur + second + pointsNum]) / 2;
            //std::cout << first << ":" << kpts[cur + first] << " , " << kpts[cur + first + pointsNum] << ";\n"
            //          << second << ":" << kpts[cur + second] << ", " << kpts[cur + second + pointsNum] << ";\n";
            points5.push_back(x);
            points5.push_back(y);
        }
        for (unsigned int j = 0; j < key_point_pos.size(); j++) {
            int idx = key_point_pos[j];
            points5.push_back(kpts[cur + idx]);
            points5.push_back(kpts[cur + idx + pointsNum]);
            //std::cout << idx << ":" << kpts[cur + idx] << " , " << kpts[cur + idx + pointsNum] << ";\n";
        }
        //std::cout << std::endl;
    }
    //show
    //std::cout << "\nget_5_points():" << points5.size() << " data: ";
    //for (unsigned int i = 0; i < points5.size(); i += 2)
    //    std::cout << points5[i] << "," << points5[i + 1] << "; ";
    //std::cout << std::endl;
}


void Affine_::get_transform(std::vector<double> &out, float *src_points, int num_boxes, int w, int h) {
    float (*src_s)[5][2] = (float (*)[5][2]) src_points;
    //std::cout << "transf ";
    for (int n = 0; n < num_boxes; n++) {
        float  (*src_)[2];
        src_ = src_s[n];
        float std_[5][2] = {{38.2946, 51.6963},
                            {73.5318, 51.5014},
                            {56.0252, 71.7366},
                            {41.5493, 92.3655},
                            {70.7299, 92.2041}};//112,112
        if (w != 112 or h != 112) {
            for (int i = 0; i < 5; i++) {
                std_[i][0] *= (w / 112.0);
                std_[i][1] *= (h / 112.0);
            }
        }
        cv::Mat std_point = cv::Mat(5, 2, CV_32FC1, std_);
        cv::Mat src_pts = cv::Mat(5, 2, CV_32FC1, src_);
        cv::Mat inliers;
        cv::Mat transformation = cv::estimateAffinePartial2D(src_pts, std_point, inliers);//cv::Mat(2, 3, CV_32FC1)
        memcpy(out.data() + 6 * n, (double *) transformation.data, 2 * 3 * sizeof(double));
//        std::cout << "\n+transformation\n" << transformation;
//        std::cout << "-inliers:" << inliers.t();
    }
//    std::cout << std::endl;
}


void Affine_::inv_transform(double *factor, float inv_coeff[][4], int num_pts) {
    double (*coeff)[2][3];
    coeff = (double (*)[2][3]) factor;
    for (int i = 0; i < num_pts; i++) {
        float cosTheta = coeff[i][0][0];
        float sinTheta = coeff[i][1][0];
        float off_x = coeff[i][0][2], _x;
        float off_y = coeff[i][1][2], _y;
        float det = pow(sinTheta, 2) + pow(cosTheta, 2);
        sinTheta /= det, cosTheta /= det;
        //求出idx，idy所在原始图像上的坐标
        _x = cosTheta * off_x + sinTheta * off_y;
        _y = -sinTheta * off_x + cosTheta * off_y;
//        std::cout << "inv coeff cossin : " << cosTheta << " "
//                  << sinTheta << " xy:" << _x << " " << _y << std::endl;
        inv_coeff[i][0] = cosTheta;
        inv_coeff[i][1] = sinTheta;
        inv_coeff[i][2] = _x;
        inv_coeff[i][3] = _y;
    }
    std::cout << "inv   coeff:";
    for (int i = 0; i < num_pts * 4; i++)std::cout << inv_coeff[i / 4][i % 4] << "; ";
    std::cout << std::endl;
}

void Affine_::static_time(long kpts5, long coeff, long invs, long kernel, long total, int box) {
    if (show_eclipse_time) {
        double a, a0, b, c, d;
        a = (double) (kpts5) / CLOCKS_PER_SEC * 1000;
        a0 = (double) (coeff) / CLOCKS_PER_SEC * 1000;
        b = (double) (invs) / CLOCKS_PER_SEC * 1000;
        c = (double) (kernel) / CLOCKS_PER_SEC * 1000;
        d = (double) (total) / CLOCKS_PER_SEC * 1000 / box;
        printf("batchs: %d with execution time: kpts: %3.4f ms | coeff: %3.4f ms | inverse: %3.4f ms | "
               "kernel: %3.4f ms | mean_total: %3.4f ms\n",
               box, a, a0, b, c, d);
    }
}

void Affine_::affineInterface(int vBatchSize, float *vp68pointsOnGPU,
                              cudaTextureObject_t vSrcImg, int vWidthSrcImg, int vHeightSrcImg,
                              float *vopDst) {
//    std::cout << "batch:" << vBatchSize << std::endl;
    clock_t start = 0, middle, middle4, middle1 = 0, middle2 = 0, middle3 = 0, end;
    int pointsNum = 68, totalNum = 73;
    int preprocess = 1, xxx = 2;
    re_malloc_coeff(vBatchSize);
    int num_pts = vBatchSize;
    std::vector<double> coeff(num_pts * 6);
    int thread_sz = (num_pts * 10 + 31) / 32 * 32;
    float inv_coeff[num_pts][4];
    start = clock();
    if (preprocess == xxx) {
        std::vector<float> pts68ss(num_pts * totalNum * 3), points5;
        int ptsSize = num_pts * totalNum * 3 * sizeof(float);//batch*68*(xyz)
        exitIfCudaError(cudaMemcpy(pts68ss.data(), vp68pointsOnGPU, ptsSize, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        std::cout << std::endl << "********************************" << std::endl;
        //for (int i = 0; i < vBatchSize; i += 1)//3*68
        //    for (int j = 33; j < 35; j += 1)//3*68
        //        std::cout << i << "." << j << ": " << pts68ss[i * totalNum * 3 + j] << "," <<
        //                  pts68ss[i * totalNum * 3 + j + pointsNum] << ","
        //                  << pts68ss[i * totalNum * 3 + j + pointsNum * 2]
        //                  << "; \n";
        //std::cout << std::endl;

        //5points
        get_5_points_general(pts68ss.data(), points5, num_pts, pointsNum, totalNum);
        //printf("-------A--------1");
        // get transformation
        get_transform(coeff, points5.data(), vBatchSize, out_w, out_h);
        //show output coeff
        std::cout << "output coeff: ";
        for (unsigned int i = 0; i < coeff.size(); i++) std::cout << coeff[i] << ", ";
        std::cout << std::endl;
        //inverse coeffection
        inv_transform(coeff.data(), inv_coeff, num_pts);
        //printf("-------A--------2");
        cudaMemcpyAsync(dev_coeff, inv_coeff, coeff_size * num_pts, cudaMemcpyHostToDevice, stream);
        //printf("--------A----- --3");
    } else {
        //  5points in gpu
        std::vector<float> points5(num_pts * 10);
        get_5kps_expand_gpu << < 1, thread_sz, 0, stream >> >
                                                  (vp68pointsOnGPU, dev_5kpts, num_pts, pointsNum, totalNum);
        middle1 = clock();
        exitIfCudaError(cudaMemcpy(points5.data(), dev_5kpts, num_pts * 10 * sizeof(float), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        // get transformation
        get_transform(coeff, points5.data(), vBatchSize, out_w, out_h);
        //show output coeff
//        std::cout << "output coeff:\n ";
//        double *xx = coeff.data();
//        for (int i = 0; i < num_pts; i++)
//            std::cout << coeff[i * 6 + 0] << ", " << coeff[i * 6 + 3] << ", " << coeff[i * 6 + 2] << ", "
//                      << coeff[i * 6 + 5] << ";\n";
//        std::cout << std::endl;
        //inverse coeffection in gpu
        cudaMemcpyAsync(coeff_params, coeff.data(), num_pts * 6 * sizeof(double), cudaMemcpyHostToDevice, stream);
        middle2 = clock();
        inv_transform_gpu << < 1, thread_sz, 0, stream >> > (coeff_params, dev_coeff, num_pts);
        middle3 = clock();
//        exitIfCudaError(cudaMemcpy(inv_coeff, dev_coeff, coeff_size * num_pts, cudaMemcpyDeviceToHost));
//        cudaDeviceSynchronize();//if use function getGpuImage(), inv_coeff is nessary
//        std::cout << "~inv   coeff:";
//        for (int i = 0; i < num_pts * 4; i++)std::cout << inv_coeff[i / 4][i % 4] << "; ";
//        std::cout << std::endl;
    }
//    printf("-------A--------4");
    //start kernel
    middle4 = clock();
//    cudaEventRecord(tic, stream);
    middle = clock();
    dim3 threads_num(128, 2);
    dim3 block_nmb(ceil(1.0f * num_pts * out_w / threads_num.x), ceil(1.0f * num_pts * out_h / threads_num.y));
    // common kernel
    ImageRotate_kernel_tex << < block_nmb, threads_num, 0, stream >> > (vSrcImg, (float3 *) vopDst,
            vWidthSrcImg, vHeightSrcImg, out_w, out_h, dev_coeff, vBatchSize);

    //record spend time
    //todo: should
//    cudaStreamSynchronize(stream);
    end = clock();

//    cudaEventRecord(toc, stream);
//    exitIfCudaError(cudaEventSynchronize(toc));
//    float time;
//    cudaEventElapsedTime(&time, tic, toc);
//    printf("kernel spend time: %3.4fms\n", time);
    static_time(middle1 - start, middle2 - middle1, middle3 - middle2, end - middle,
                end - middle + middle4 - start, num_pts);
//    getGpuImage(vBatchSize, vopDst, inv_coeff);
}


void Affine_::getGpuImage(int num_boxs, float *dev_img_out, float inv_coeff[][4]) {
    cv::Mat imgCroped(out_h, out_w, CV_32FC3, cv::Scalar(0, 0, 0)), dst;
    cv::Mat tmp1(out_h, out_w, CV_32FC1);
    cv::Mat tmp2(out_h, out_w, CV_32FC1);
    cv::Mat tmp3(out_h, out_w, CV_32FC1);
    int nImg = out_h * out_w;
    int NumPixels = out_h * out_w * 3;
    float *pDevice = dev_img_out;
    bool CHW = true;//    ;"HWC"
    for (int i = 0; i < num_boxs; i++) {
        if (not CHW) {
            exitIfCudaError(cudaMemcpy(imgCroped.data, pDevice,
                                       NumPixels * sizeof(float), cudaMemcpyDeviceToHost));
            pDevice += NumPixels;
        } else {
            exitIfCudaError(cudaMemcpy(tmp1.data, pDevice, nImg * sizeof(float), cudaMemcpyDeviceToHost));
            pDevice += nImg;
            exitIfCudaError(cudaMemcpy(tmp2.data, pDevice, nImg * sizeof(float), cudaMemcpyDeviceToHost));
            pDevice += nImg;
            exitIfCudaError(cudaMemcpy(tmp3.data, pDevice, nImg * sizeof(float), cudaMemcpyDeviceToHost));
            pDevice += nImg;
            std::vector <cv::Mat> mbgr = {tmp1, tmp2, tmp3};
            cv::merge(mbgr, imgCroped);
        }

        cudaDeviceSynchronize();
        imgCroped.convertTo(dst, CV_8UC3);
//        std::cout << "imgcroped tyep:" << dst.type() << "," << dst.rows << "," <<
//                  dst.cols << "save:" << std::endl;
        std::cout << "inv   coeff:";
        for (int j = 0; j < 4; j++)std::cout << inv_coeff[i][j] << "; ";
        std::cout << std::endl;

        char *ch = get_random_name();
        cv::imwrite(ch, dst);

    }
}

char *Affine_::get_random_name() {
    const char *ch = "/home/lidachong/project/VisionProject/results/";
    const char CCH[] = "_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_";
//    const int SIZE_CHAR = 32;
    char pre_ch[] = "01234.jpg";
    char *ch3 = (char *) malloc(strlen(ch) + strlen(pre_ch) + 1);
    for (int i_ch = 0; i_ch < 5; i_ch++) {
        int x = rand() / (RAND_MAX / (sizeof(CCH) - 1));
        pre_ch[i_ch] = CCH[x];
    }
    for (int i = 0; ch[i] != '\0'; i++)
        ch3[i] = ch[i];
    for (int i = 0; pre_ch[i] != '\0'; i++)
        ch3[i + strlen(ch)] = pre_ch[i];
    ch3[strlen(ch) + strlen(pre_ch)] = '\0';
    for (int i = 0; pre_ch[i] != '\0'; i++)printf("%c", pre_ch[i]);
    printf("  - \n");
    return ch3;
}
