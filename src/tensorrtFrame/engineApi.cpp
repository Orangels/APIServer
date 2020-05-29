#include <unordered_map>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "cudaUtility.h"
#include "tensorNet.h"
#include "FileFunction.h"
#include "engineApi.h"
#include "engine_utils.h"

Engine_Api ::Engine_Api()
{
    m_pconfiger = CConfiger::getOrCreateConfiger();
    std::string g_modelPrefix,point_g_modelPrefix,ageGender_g_modelPrefix;

    g_modelPrefix = m_pconfiger->readValue("ssd_detection_modelPrefix");
    initialize(g_modelPrefix.c_str(), "ssdInference");

    point_g_modelPrefix = m_pconfiger->readValue("point_modelPrefix");
    initialize(point_g_modelPrefix.c_str(), "keyPointInference");

    ageGender_g_modelPrefix = m_pconfiger->readValue("ageGender_modelPrefix");
    initialize(ageGender_g_modelPrefix.c_str(), "ageGenderInference");
}


Engine_Api::~Engine_Api()
{

}

void Engine_Api::detect_headface(cv::Mat &image, std::vector<int>& hf_boxs)
{
    cv::Mat faceImg = image;
    std::vector<uchar> imgs;
    int numDetection = 0, nDets[64];
    float box_arr[120];
	float* box = box_arr;
    int g_batchSize = m_pconfiger->readValue<int>("batchSize");

    trans_date(faceImg, imgs, g_batchSize);

    numDetection = ssdInference(g_batchSize, faceImg.rows, faceImg.cols, imgs.data(), box, nDets/*reFace.data*/);

    for (int j = 0; j < numDetection; ++j) {
        for(int i =0; i< 6;i++){
            hf_boxs.push_back(int(box[i]));
        }
		box += 6;
	}
}

void Engine_Api::detect_headface(cv::Mat &image, void* box)
{
    cv::Mat faceImg = image;
    std::vector<uchar> imgs;
    int numDetection = 0, nDets[64];
    float box_arr[120];
	box = box_arr;
    int g_batchSize = m_pconfiger->readValue<int>("batchSize");

    trans_date(faceImg, imgs, g_batchSize);

    numDetection = ssdInference(g_batchSize, faceImg.rows, faceImg.cols, imgs.data(), box, nDets/*reFace.data*/);
}

void Engine_Api::get_angles(cv::Mat &image, std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& angles)
{
	cv::Mat img = image.clone();
    img.convertTo(img, CV_32FC3);
    int model_batch = 8;
    int num_face = rects.size();
    std::vector<int> data;
	for(auto& rect : rects){
	    data.insert(data.end(), rect.begin(),rect.end());
	}
	std::vector<float> r(num_face * (62 + 219 + 4));

	keyPointInference(num_face, data.data(), r.data());
	float * ret = r.data();
	for (int j = 0; j < num_face; ++j) {
	    std::vector<float> angle;
	    float Y,P,R;
	    Y = *(ret+62*num_face + 216 +j*219);
	    P = *(ret+62*num_face + 217 +j*219);
	    R = *(ret+62*num_face + 218 +j*219);

	    angle.push_back(Y);
	    angle.push_back(P);
	    angle.push_back(R);
	    angles.push_back(angle);
	}
}

void Engine_Api::get_angles(cv::Mat &image, std::vector<std::vector<int>>& rects, std::vector<float>& angles)
{
    cv::Mat img = image.clone();
    img.convertTo(img, CV_32FC3);
    int model_batch = 8;
    int num_face = rects.size();
    std::vector<int> data;
    for(auto& rect : rects){
        data.insert(data.end(), rect.begin(),rect.end());
    }
    std::vector<float> r(num_face * (62 + 219 + 4));

    keyPointInference(num_face, data.data(), r.data());
    angles = r;
}

void Engine_Api::get_angles(cv::Mat &image, std::vector<std::vector<int>>& rects, void * angles)
{
	cv::Mat img = image.clone();
    img.convertTo(img, CV_32FC3);
    int model_batch = 8;
    int num_face = rects.size();
    std::vector<int> data;
	for(auto& rect : rects){
	    data.insert(data.end(), rect.begin(),rect.end());
	}
	std::vector<float> r(num_face * (62 + 219 + 4));
	angles = r.data();

	keyPointInference(num_face, data.data(), r.data());

}

void Engine_Api::get_ageGender(cv::Mat &image, std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& infos)
{
    int vw = 112, vh = 112;
    int num_face = rects.size();
    std::vector<float> r(num_face*515);//64, 56, 56
	std::vector<int> data;
	for(auto& rect : rects){
		data.insert(data.end(), rect.begin(), rect.end());
	}

    ageGenderInference(num_face, data.data(), r.data());
    float * ret = r.data();
    for (int j = 0; j < num_face; ++j) {
        std::vector<float> info;
        for(int i = 0; i < 515; i++){
            info.push_back(*(ret+515*j + i));
        }
        infos.push_back(info);
    }
}

void Engine_Api::get_ageGender(cv::Mat &image, std::vector<std::vector<int>>& rects, std::vector<float>& infos)
{
    int vw = 112, vh = 112;
    int num_face = rects.size();
    std::vector<float> r(num_face*515);//64, 56, 56
    std::vector<int> data;
    for(auto& rect : rects){
        data.insert(data.end(), rect.begin(), rect.end());
    }

    ageGenderInference(num_face, data.data(), r.data());
    infos = r;
}

void Engine_Api::get_ageGender(cv::Mat &image, std::vector<std::vector<int>>& rects, void * infos)
{
    int vw = 112, vh = 112;
    int num_face = rects.size();
    std::vector<float> r(num_face*515);//64, 56, 56
	std::vector<int> data;
	for(auto& rect : rects){
		data.insert(data.end(), rect.begin(), rect.end());
	}
    infos = r.data();
    ageGenderInference(num_face, data.data(), r.data());
}