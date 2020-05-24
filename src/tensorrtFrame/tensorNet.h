//#include "pluginImplement.h"
#ifndef __TENSOR_NET_H__
#define __TENSOR_NET_H__
#include "ImagePreprocessor.h"
#include "affineInterface.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include <iostream>
using namespace nvinfer1;
using namespace nvcaffeparser1;


template<typename T>
double getSum(T* p, int n, bool vAbs = true, bool vShowFirstElement= false)
{
    if(vShowFirstElement)
        std::cout << "getSum[0]: " << *p << std::endl;
    double sumN = 0;
    while (n-- > 0)
    {
//        std::cout << "n : "<< n << std::endl;
        T temp = *p++;
        if (vAbs && temp < 0) temp *= -1;
        sumN += temp;
    }

    std::cout << "sumN : "<< sumN << std::endl;
    return sumN;
}

/******************************/
// TensorRT utility
/******************************/
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        if (severity!=Severity::kINFO) std::cout << msg << std::endl;
    }
};

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });

        if (record == mProfile.end()) mProfile.push_back(std::make_pair(layerName, ms));
        else record->second += ms;
    }

    void printLayerTimes(const int TIMING_ITERATIONS)
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
    }
};


/******************************/
// TensorRT Main
/******************************/
namespace cv
{
    class Mat;
}
class CCaffePrototxtReader;
class TensorNet
{
public:
    bool LoadNetwork(const std::string& vModelPrefix, uint32_t vMaxBatchSize, cudaStream_t vpCudaStream=NULL );

    DimsCHW allocateIOBuffer(size_t voBufferSize = 0);
    void imageInference(int vBatchSize, const std::string& vName="");
    DimsCHW getTensorDims(const char* name);

    Affine_ affine_instance;
    void printTimes(int iteration);
    void destroy();
    std::vector<void*> modelIObuffers;
    float* resultBufferHost = NULL;
    size_t modelOutputBytes = 0;
    size_t modelInputBytes = 0;
    cudaStream_t cudaStream;
    cudaEvent_t startCuda, endCuda;
    DimsCHW m_dimsOut, m_indims;
	int m_maxBatchSize = 1;
    CImagePreprocessor* imagePreprocessor=NULL;
    int caculateDetail3Dkeypoints(int vBatchSize, int* vpBoxes);
    void postSSD(int vBatchSize, int vHeight, int vWidth);
    void affine(int vBatchSize, float* vp68points);
    ~TensorNet();
	ICudaEngine* m_pEngine;

private:
    int __getSSDdetections(float vConfidence);
    void __createInferenceEngine(void* vpPlanFileData, size_t vBites);
    bool __caffeToTRTModel(std::string vCaffeModelPrefix);
	void __saveModelCache2File(const std::string& vCacheFileName, const IHostMemory* vpModelCache, int vMaxBatchSize);
	void __creatTrtModelFromCache(const std::string& vCacheFileName);
	bool __isSSD();
//	bool __onnx2TRTModel(const std::string& vOnnxFileName);
//    PluginFactory m_pluginFactory;
    IHostMemory *m_pGieModelStream{nullptr};
	CCaffePrototxtReader* m_pCaffeReader;

    IRuntime* m_pInfer;
    cudaTextureObject_t m_shpBase, m_expBase;
    CCuda3DArray* m_pshpBase=NULL, *m_pexpBase=NULL;
    Logger m_logger;
    Profiler m_profiler;
    IExecutionContext* m_pContext = NULL;
    bool m_fp16 = true;
    size_t m_iter = 0;
	std::vector<float> m_buffer;
};


#endif

