#include <unordered_map>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "cudaUtility.h"
#include "tensorNet.h"
#include "Common.h"
#include "FileFunction.h"
#include "detection.h"


std::unordered_map<int, std::unordered_map<std::string, TensorNet>> g_trtNetSet(8);
TensorNet* g_ptensor = NULL;
cudaStream_t g_cudaStream = NULL;
DimsCHW g_outputDim;
int  g_gpuindex = 0;
int g_batchSize = 1;
CTimer g_taskTimer, g_inferTimer;


std::vector<int> get_dim_size(Dims dim)
{
	std::vector<int> size;
	for (int i = 0; i < dim.nbDims; ++i)
		size.emplace_back(dim.d[i]);
	return size;
}

int total_size(std::vector<int> dim)
{
	int size = 1 * sizeof(float);
	for (auto d : dim)
		size *= d;
	return size;
}

template<typename T>
void loadImage(int vHeight, int vWidth, T* vsrc, float* dst, int vBatchSize)
{
	/*
	PIXEL_MEANS: [103.52, 116.28, 123.675] #BGR order as well as cv2.imread
	PIXEL_STDS: [57.375, 57.12, 58.395]
	*/
	T* src = vsrc;
	unsigned int numPixel = vHeight * vWidth;
	float *p0 = dst, *p1 = dst + numPixel, *p2 = p1 + numPixel;
	while (numPixel--)
	{
		*p0 = (*src); p0++; src++;/*/ 255.f; src     */
		*p1 = (*src); p1++; src++;/*/ 255.f; src     */
		*p2 = (*src); p2++; src++;/*/ 255.f; src     */
	}
	unsigned int numElement = vHeight * vWidth * 3;
	for (int i = 1; i < vBatchSize; ++i)
	{
		memcpy(dst + numElement*i, dst, sizeof(T)*numElement);
	}
}

struct LayerInfo
{
	std::vector<int> dim;
	std::string name;
	int index;
	int size;
};
std::vector<LayerInfo> output_layer;
void _outputSSD(float xMin, float yMin, float xMax, float yMax, float vConfidence, int vClassID)
{
	std::cout << vClassID << ", " << vConfidence << " : " << xMin << ", " << yMin << ", " << xMax << ", " << yMax << std::endl;
}
void setMaxCameraNum(int vMaxThreadNum) { g_trtNetSet.reserve(vMaxThreadNum * 2); }
void initialize(const char* vpModelPrefixFileName, char* vpFunctionName)
{
    int maxBatchSize = g_batchSize; //it is valid only when creating tensorcache file and maxBatchSize>conterpart of model file.
    cudaSetDevice(g_gpuindex);

    if (g_trtNetSet.bucket_count() < 4) setMaxCameraNum(4);
    std::cout << "***"<<std::endl;
    int threadId = getThreadID();
    std::cout << threadId << std::endl;
	std::cout << "***"<<std::endl;
    std::string funName(vpFunctionName);
    TensorNet& tensorNet = g_trtNetSet[threadId][funName];
    std::string prefix(vpModelPrefixFileName);
    //if (isEndWith(prefix, "onnx")) maxBatchSize = 16;
    tensorNet.LoadNetwork(prefix, maxBatchSize, g_cudaStream);
    g_cudaStream = tensorNet.cudaStream;
    if (funName == "keyPointInference")
    {
        g_outputDim = tensorNet.allocateIOBuffer((62 + 73 * 3 + 4) * sizeof(float));//4 for box rect for development
        std::unordered_map<std::string, TensorNet>& threadModels = g_trtNetSet[threadId];
        if (threadModels.find("ssdInference") != threadModels.cend()){
			tensorNet.imagePreprocessor = threadModels["ssdInference"].imagePreprocessor;
			std::cout << "find ssd Inference"<< std::endl;
        }
        else tensorNet.imagePreprocessor = new CImagePreprocessor;
        tensorNet.imagePreprocessor->setSizeResizedImage(tensorNet.m_indims.w(), tensorNet.m_indims.h(), tensorNet.m_maxBatchSize, true);
        tensorNet.imagePreprocessor->setOutput((float*)tensorNet.modelIObuffers[0], true);
    }
    else
    {
        if (funName == "ageGenderInference")
        {
            std::unordered_map<std::string, TensorNet>& threadModels = g_trtNetSet[threadId];
            tensorNet.imagePreprocessor = threadModels["keyPointInference"].imagePreprocessor;
        }
        g_outputDim = tensorNet.allocateIOBuffer(0);
    }

    if (funName == "retinaFace")
    {
        auto& engine = tensorNet.m_pEngine;
        for (int b = 0; b < engine->getNbBindings(); ++b)
        {
            if (!engine->bindingIsInput(b))
            {
                LayerInfo l;
                l.name = engine->getBindingName(b);
                Dims dim_output = engine->getBindingDimensions(b);
                l.dim = get_dim_size(dim_output);
                l.size = total_size(l.dim);
                l.index = b;
                output_layer.emplace_back(l);
            }
        }


    }
    std::cout << "RT init done!" << std::endl;
}

int _inference(int vBatchSize, int* vpImgData, void* vopPoint, const std::string& vFunctionName)
{
	int threadId = getThreadID();
	TensorNet& tensorNet = g_trtNetSet[threadId][vFunctionName];
	if ("keyPointInference" == vFunctionName && vBatchSize > 0)
		tensorNet.imagePreprocessor->cropResize(vBatchSize, (int*)vpImgData, tensorNet.cudaStream);
	else if ("ageGenderInference" == vFunctionName && vBatchSize > 0)
	{
		float* p68 = (float*)g_trtNetSet[threadId]["keyPointInference"].modelIObuffers.back();
		p68 += 62 * vBatchSize;
		tensorNet.affine(vBatchSize, p68);
	}
	else cudaMemcpyAsync(tensorNet.modelIObuffers[0], vpImgData, std::abs(vBatchSize)*tensorNet.modelInputBytes, cudaMemcpyHostToDevice, tensorNet.cudaStream);
	tensorNet.imageInference(std::abs(vBatchSize), vFunctionName);
	if ("keyPointInference" == vFunctionName)
		tensorNet.caculateDetail3Dkeypoints(std::abs(vBatchSize), vBatchSize>0 ? (int*)vpImgData : (int*)vopPoint);
	cudaStreamSynchronize(tensorNet.cudaStream);
	float* output = (float*)tensorNet.resultBufferHost;
//	for(int i =0;i<285;i++){
//		std::cout<<i<<"="<<*(output+i)<<" ";
//	}
	auto& d = tensorNet.m_dimsOut;
	size_t bites = d.c()*d.w()*d.h();
//	std::cout << "ls bites : " << bites << std::endl;
//	if ("keyPointInference" == vFunctionName) bites += 219;
	if ("keyPointInference" == vFunctionName) bites = 285;
	bites *= std::abs(vBatchSize) * sizeof(float);
//	vopPoint = output;
	memcpy(vopPoint, output, bites);
//    for(int i =0;i<bites/sizeof(float);i++){
//		std::cout<<i<<"="<<*(output+i)<<" ";
//	}
//	printf("cpp dst sum = %.2f\n", getSum((float*)output, std::abs(vBatchSize) * d.c()*d.w()*d.h()));
	return std::abs(vBatchSize);
}

int _inference(int vBatchSize, float* vpImgData, void* vopPoint, const std::string& vFunctionName)
{

	int threadId = getThreadID();
	TensorNet& tensorNet = g_trtNetSet[threadId][vFunctionName];
	if ("keyPointInference" == vFunctionName && vBatchSize > 0)
		tensorNet.imagePreprocessor->cropResize(vBatchSize, (int*)vpImgData, tensorNet.cudaStream);
	else if ("ageGenderInference" == vFunctionName && vBatchSize > 0)
	{
		float* p68 = (float*)g_trtNetSet[threadId]["keyPointInference"].modelIObuffers.back();
		p68 += 62 * vBatchSize;
		tensorNet.affine(vBatchSize, p68);
	}
	else cudaMemcpyAsync(tensorNet.modelIObuffers[0], vpImgData, std::abs(vBatchSize)*tensorNet.modelInputBytes, cudaMemcpyHostToDevice, tensorNet.cudaStream);
	tensorNet.imageInference(std::abs(vBatchSize), vFunctionName);
	if ("keyPointInference" == vFunctionName)
		tensorNet.caculateDetail3Dkeypoints(std::abs(vBatchSize), vBatchSize>0 ? (int*)vpImgData : (int*)vopPoint);
	cudaStreamSynchronize(tensorNet.cudaStream);
	float* output = (float*)tensorNet.resultBufferHost;
//	for(int i =0;i<219;i++){
//	    std::cout<<i<<"="<<*(output+i)<<" ";
//	}
	auto& d = tensorNet.m_dimsOut;
	size_t bites = d.c()*d.w()*d.h();
	if ("keyPointInference" == vFunctionName) bites += 219;
	bites *= std::abs(vBatchSize) * sizeof(float);
//	vopPoint = output;
	memcpy(vopPoint, output, bites);
	printf("cpp dst sum = %.2f\n", getSum((float*)output, std::abs(vBatchSize) * d.c()*d.w()*d.h()));
	return std::abs(vBatchSize);
}
int keyPointInference(int vBatchSize, int* vpImgData, void* vopPoint)
{
    return _inference(vBatchSize, vpImgData, vopPoint, "keyPointInference");
//	return ls_inference(vBatchSize, vpImgData, vopPoint, "keyPointInference");
}
int ageGenderInference(int vBatchSize, int* vpImgData, void* vopPoint)
{
    return _inference(vBatchSize, vpImgData, vopPoint, "ageGenderInference");
}

int verifyFaceInference(int vBatchSize, float* vpImgData, void* vopPoint)
{
    return _inference(vBatchSize, vpImgData, vopPoint, "verifyFaceInference");
}

int verifyFaceInference2(int vBatchSize, float* vpImgData, void* vopPoint)
{
    g_cudaStream = NULL;
    return _inference(vBatchSize, vpImgData, vopPoint, "verifyFaceInference2");
}
int ssdInference(int vBatchSize, int vHeight, int vWidth, uchar* vpImgData,
		void* vopBoxScoreCls, int* vopNumDetections)
	{
		TensorNet& tensorNet = g_trtNetSet[getThreadID()]["ssdInference"];
		//int* pBox = (int*)vopBoxScore;
		float* pBoxScoreCls = (float*)vopBoxScoreCls;
		std::vector<cv::Mat> images(vBatchSize);
		int imgSize = vHeight * vWidth * 3;
		for (auto& image : images)
		{
			image = cv::Mat(vHeight, vWidth, CV_8UC3, vpImgData);
			vpImgData += imgSize;
		}
		tensorNet.imagePreprocessor->preprocess(images, tensorNet.cudaStream);
		g_inferTimer.start();
		tensorNet.imageInference(vBatchSize, "ssdInference");
		tensorNet.postSSD(vBatchSize, vHeight, vWidth);
		// 		if (0 != vopImage)
		// 			tensorNet.imagePreprocessor->getResizedImage((float*)tensorNet.modelIObuffers[0], (char*)vopImage, caffeReader.getDim()[3], caffeReader.getDim()[2], tensorNet.cudaStream);
		g_inferTimer.stop();
		g_ptensor = &tensorNet;

		auto& d = tensorNet.m_dimsOut;
		static int iter = 0; //iter++;
		int numAll = 0;
		for (int i = 0; i < vBatchSize; ++i)
		{
			float* output = (float*)tensorNet.resultBufferHost + i*tensorNet.modelOutputBytes / sizeof(float);
			int numDetection = *output, tnum = 0;
			for (int k = 0; k < numDetection; k++)
			{
				if (output[1] >= 0)
				{
					tnum++;
					memcpy(pBoxScoreCls, output + 3, 4 * sizeof(float));
					pBoxScoreCls[4] = output[2];
					pBoxScoreCls[5] = output[1];
//					if (1 == iter % 50) _outputSSD(output[3], output[4], output[5], output[6], output[2], output[1]);
//					_outputSSD(output[3], output[4], output[5], output[6], output[2], output[1]);
					pBoxScoreCls += 6;
				}
				output += 7;
			}
			*vopNumDetections++ = tnum;
			numAll += tnum;
		}
		return  numAll;
	}

SSD_Detection ::SSD_Detection()
{
    std::cout <<"init"<<std::endl;
    m_pconfiger = CConfiger::getOrCreateConfiger();
    std::string g_modelPrefix,point_g_modelPrefix,ageGender_g_modelPrefix;
    g_modelPrefix = m_pconfiger->readValue("ssd_detection_modelPrefix");
    point_g_modelPrefix = m_pconfiger->readValue("point_modelPrefix");
    initialize(g_modelPrefix.c_str(), "ssdInference");
//    std::cout <<g_modelPrefix<<std::endl;

    initialize(point_g_modelPrefix.c_str(), "keyPointInference");
//    std::cout <<point_g_modelPrefix<<std::endl;
    ageGender_g_modelPrefix = m_pconfiger->readValue("ageGender_modelPrefix");
    initialize(ageGender_g_modelPrefix.c_str(), "ageGenderInference");
}

SSD_Detection::~SSD_Detection()
{

}


void SSD_Detection::detect_hf(cv::Mat image, std::vector<int>& hf_boxs)
//void SSD_Detection::detect_hf()
{
//    std::cout <<"detect_hf"<<std::endl;
    std::vector<float> r;
	float box_arr[120];
	float* box = box_arr;
    cv::Mat faceImg = image;

	int n = faceImg.cols*faceImg.rows*faceImg.channels();
	std::vector<uchar> imgs(n*g_batchSize);
	for (int i = 0; i < g_batchSize; ++i)
	{
		memcpy(imgs.data() + i*n, faceImg.data, n);
	}
	int numDetection = 0, nDets[64];
	numDetection = ssdInference(g_batchSize, faceImg.rows, faceImg.cols, imgs.data(), box, nDets/*reFace.data*/);

//	std::cout << getThreadID() << " numDetection: " << numDetection << ": ";
//	for (int i = 0; i < g_batchSize; ++i) std::cout << nDets[i] << ", "<< std::endl;
	for (int j = 0; j < numDetection; ++j) {
//		std::cout << "********" << std::endl;
//		std::cout << box[0]<< " : " << box[1] << " , " << box[2] << " , " << box[3] << " , " << box[4] << " , " << box[5] << " , " << std::endl;
		hf_boxs.push_back(int(box[0]));
		hf_boxs.push_back(int(box[1]));
		hf_boxs.push_back(int(box[2]));
		hf_boxs.push_back(int(box[3]));
		hf_boxs.push_back(int(box[4]));
		hf_boxs.push_back(int(box[5]));
		box += 6;
	}


}


void SSD_Detection::get_angles(cv::Mat image, std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& angles)
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
	    for(int i = 0; i < 68; i++){
	        cv::circle(img,cv::Point(int(*(ret+i+62*num_face+j*219)),int(*(ret+i+62*num_face+j*219+68))),3,(0,0,213),2);
	    }
	    std::vector<float> angle;
	    float Y,P,R;
	    Y = *(ret+62*num_face + 216 +j*219);
	    P = *(ret+62*num_face + 217 +j*219);
	    R = *(ret+62*num_face + 218 +j*219);

//	    std::cout <<" Y:"<<Y<<" P:"<<P<<" R:"<<R<<std::endl;
	    angle.push_back(Y);
	    angle.push_back(P);
	    angle.push_back(R);
	    angles.push_back(angle);
	}

	/*
    std::cout << "num_face : " << num_face << std::endl;
    int points_batch_num = ceil(num_face / float(model_batch));
    std::cout << "points_batch_num : " << points_batch_num << std::endl;

    for(int t =0;t<points_batch_num; t++){
        int current_size = (t+1) * model_batch > num_face ? (num_face%model_batch) : model_batch;
        std::vector<int> data_batch;
        for(int i = 0;i<current_size; i++){
            data_batch.push_back(data[(t*model_batch + i)*4]);
            data_batch.push_back(data[(t*model_batch + i)*4 + 1]);
            data_batch.push_back(data[(t*model_batch + i)*4 + 2]);
            data_batch.push_back(data[(t*model_batch + i)*4 + 3]);
        }
	    std::vector<float> r(current_size * (62 + 219 + 4));
        keyPointInference(current_size, data_batch.data(), r.data());
	    float * ret = r.data();
	    for (int j = 0; j < current_size; ++j) {
    //    for (int j = 0; j < 1; ++j) {
		    for(int i = 0; i < 68; i++){
//			cv::circle(img,cv::Point(int(*(ret+i+62 +62*j + 68*j)),int(*(ret+i+62+62*j+68+68*j))),3,(0,0,213),2);
			cv::circle(img,cv::Point(int(*(ret+i+62*current_size+j*219)),int(*(ret+i+62*current_size+j*219+68))),3,(0,0,213),2);
			std::vector<float> angle;
			*(ret+i+62*current_size+j*219)
			angle.push_back();
			angle.push_back();
			angle.push_back();
			angles.push_back(angle);
		    }
	    }
    }
    */

//	std::vector<int> data = {297, 222, 521, 446};

//	float data[4] = {297, 222, 521, 446};
//	loadImage(vw, vh, (float*)img.data, data.data(), g_batchSize);
//	std::vector<float> r(g_batchSize * (62 + 209));

//	float r[281];
//	int face[] = { 0,0, 120, 120 };
//	for (int k = 0; k < g_batchSize; ++k)
//	{
//		memcpy(r.data() + 4 * k, face, sizeof(face));
//	}

//	cv::imwrite("face_points.jpg",img);
//	printf("%d, ageGenderInference result sumabs = %.2f", g_batchSize, getSum(r.data(), r.size()));
}

void SSD_Detection::get_ageGender(cv::Mat image, std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& infos)
{
    int vw = 112, vh = 112;
	std::vector<float> r(64*56*56);//64, 56, 56
    int num_face = rects.size();
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
//        std::cout<< "age"<< info[512];
        infos.push_back(info);

    }

//    printf("%d, ageGenderInference result sumabs = %.2f", g_batchSize, getSum(r.data(), r.size()));
}

void SSD_Detection::get_features(std::vector<std::vector<int>>& rects, std::vector<std::vector<float>>& features)
{

}

