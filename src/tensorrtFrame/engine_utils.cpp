
#include "engine_utils.h"

std::unordered_map<int, std::unordered_map<std::string, TensorNet>> g_trtNetSet(8);
TensorNet* g_ptensor = NULL;
std::unordered_map<int,cudaStream_t> g_cudaStream(2);
//cudaStream_t g_cudaStream = NULL;
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
    int threadId = getThreadID();
    std::cout << threadId << std::endl;
    std::string funName(vpFunctionName);
    TensorNet& tensorNet = g_trtNetSet[threadId][funName];
    std::string prefix(vpModelPrefixFileName);
    //if (isEndWith(prefix, "onnx")) maxBatchSize = 16;
    tensorNet.LoadNetwork(prefix, maxBatchSize, g_cudaStream[getThreadID()]);
    g_cudaStream[getThreadID()] = tensorNet.cudaStream;
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

	auto& d = tensorNet.m_dimsOut;
	size_t bites = d.c()*d.w()*d.h();

	if ("keyPointInference" == vFunctionName) bites = 285;
	bites *= std::abs(vBatchSize) * sizeof(float);

	memcpy(vopPoint, output, bites);

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

	auto& d = tensorNet.m_dimsOut;
	size_t bites = d.c()*d.w()*d.h();
	if ("keyPointInference" == vFunctionName) bites += 219;
	bites *= std::abs(vBatchSize) * sizeof(float);

	memcpy(vopPoint, output, bites);
	printf("cpp dst sum = %.2f\n", getSum((float*)output, std::abs(vBatchSize) * d.c()*d.w()*d.h()));
	return std::abs(vBatchSize);
}
int keyPointInference(int vBatchSize, int* vpImgData, void* vopPoint)
{
    return _inference(vBatchSize, vpImgData, vopPoint, "keyPointInference");

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
    g_cudaStream[getThreadID()] = NULL;
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
//                _outputSSD(output[3], output[4], output[5], output[6], output[2], output[1]);
                pBoxScoreCls += 6;
            }
            output += 7;
        }
        *vopNumDetections++ = tnum;
        numAll += tnum;
    }
    return  numAll;
}


void trans_date(cv::Mat & faceImg, std::vector<uchar> &imgs, int batch_size)
{
    int n = faceImg.cols*faceImg.rows*faceImg.channels();
    imgs.resize(n * batch_size);
    for (int i = 0; i < batch_size; ++i)
	{
		memcpy(imgs.data() + i*n, faceImg.data, n);
	}
}
