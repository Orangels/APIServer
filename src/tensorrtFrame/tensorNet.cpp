#include <algorithm>
#include <sstream>
#include <fstream>
#include <opencv2/highgui/highgui.hpp>
#include "tensorNet.h"
#include "FileFunction.h"
//#include "caffePlugin.h"
#include "Common.h"
#include "cudaUtility.h"
#include <unordered_map>
#include <algorithm>
#include <numeric>
using namespace nvinfer1;
float* allocateLockedCPUMemory(size_t vBities)
{
    float* ptr;
    assert(!cudaMallocHost(&ptr, vBities));
    return ptr;
}


bool TensorNet::LoadNetwork(const std::string& vModelPrefix, uint32_t vMaxBatchSize, cudaStream_t vpCudaStream/*=NULL */)
{
	m_maxBatchSize = vMaxBatchSize;
    std::string cacheFileName = deleteExtentionName(vModelPrefix) +"tensorcache";
	std::cout << "attempting to open cache file " << cacheFileName << std::endl;
    std::cout << "initLibNvInferPlugins..." << std::endl;
    initLibNvInferPlugins(&m_logger,"");
	if (isExist(cacheFileName)) __creatTrtModelFromCache(cacheFileName);
    else 
    {
		std::cout << "cache file not found, creating tensorRT model form " << vModelPrefix << "*." << std::endl;
		__caffeToTRTModel(vModelPrefix);
//		if (false == flagConvertOk){std::cout << "failed to convert model from  " << vModelPrefix << "*." << std::endl; return false;}
		__createInferenceEngine(m_pGieModelStream->data(), m_pGieModelStream->size());
    }

    if (NULL == vpCudaStream) cudaStreamCreate(&cudaStream);
    else cudaStream = vpCudaStream;
    return true;
}

bool TensorNet::__caffeToTRTModel(std::string vCaffeModelPrefix)
{
	std::string  netFileName(vCaffeModelPrefix + ".prototxt");
	std::string  weightFileName(vCaffeModelPrefix+".caffemodel");
    std::string outBlobName;

    int index = vCaffeModelPrefix.find("SSD");
    bool isDetetionModel = (index >= 0 && index < vCaffeModelPrefix.size());
    if(isDetetionModel) {
        outBlobName = "detection_out";
        m_maxBatchSize = 1;
    }
    else {
        outBlobName = "out";
        m_maxBatchSize = 8;
    }
    std::cout<<outBlobName<<"  "<<vCaffeModelPrefix<<std::endl;

    IBuilder* builder = createInferBuilder(m_logger);
    //m_fp16 = 1;
    if (m_fp16) m_fp16 = builder->platformHasFastFp16();
    if (m_fp16)
    {
        //builder->setHalf2Mode(true);
        builder->setFp16Mode(true);
        std::cout << "use float16" << std::endl;
    }
//    builder->setMinFindIterations(3);
//    builder->setAverageFindIterations(2);
    builder->setMaxBatchSize(m_maxBatchSize);
    builder->setMaxWorkspaceSize(512 << 20);

    ICaffeParser* caffeParser = createCaffeParser();
    DataType modelDataType = m_fp16 ? DataType::kHALF : DataType::kFLOAT;
    INetworkDefinition* network = builder->createNetwork();

    std::cout << netFileName.c_str() << std::endl  << weightFileName.c_str() << std::endl;
    const IBlobNameToTensor* blobNameToTensor = caffeParser->parse(netFileName.c_str(), weightFileName.c_str(), *network, modelDataType);
    std::cout << "parse done " << std::endl;

    assert(blobNameToTensor != nullptr);
    network->markOutput(*blobNameToTensor->find(outBlobName.c_str()));
    std::cout << network << std::endl;
    std::cout <<"buildCudaEngineing:" << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "finished" << std::endl;
    assert(engine);
    network->destroy();
    caffeParser->destroy();

    m_pGieModelStream = engine->serialize();
    if(!m_pGieModelStream)
    {
        std::cout << "failed to serialize CUDA engine" << std::endl;
        return false;
    }
	__saveModelCache2File(vCaffeModelPrefix + ".tensorcache", m_pGieModelStream, m_maxBatchSize);
	engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
    std::cout << "caffeToTRTModel Finished" << std::endl;
    return true;
}


void TensorNet::__saveModelCache2File(const std::string& vCacheFileName, const IHostMemory* vpModelCache, int vMaxBatchSize)
{
	std::ofstream Fout(vCacheFileName, std::ios::binary);
	if (Fout.fail()) { std::cout << "Fail to wirte data to file " << vCacheFileName << std::endl; return ; }
	Fout.write((const char*)vpModelCache->data(), vpModelCache->size());
	Fout.write((const char*)&vMaxBatchSize, sizeof(vMaxBatchSize));
	Fout.close();
}


void TensorNet::__creatTrtModelFromCache(const std::string& vCacheFileName)
{
    std::cout << "initLibNvInferPlugins..." << std::endl;
    initLibNvInferPlugins(&m_logger,"");
	std::cout << "loading network profile from cache..." << std::endl;
	size_t modelSize = 0;
	void* pBuffer = NULL;
	readBinaryDataFromFile2Memory(vCacheFileName, pBuffer, modelSize);
	modelSize -= sizeof(m_maxBatchSize);
	__createInferenceEngine(pBuffer, modelSize);
	memcpy(&m_maxBatchSize, (char*)pBuffer + modelSize, sizeof(m_maxBatchSize));
	free(pBuffer);
	std::cout << "batchSize from cache: " << m_maxBatchSize << std::endl;
}

bool TensorNet::__isSSD()
{
    return false;
}

void TensorNet::__createInferenceEngine(void* vpPlanFileData, size_t vBites)
{
    m_pInfer = createInferRuntime(m_logger);
    m_pEngine = m_pInfer->deserializeCudaEngine(vpPlanFileData, vBites, nullptr);
    std::cout << "createExecutionContext vBites:" << vBites << std::endl;
    m_pEngine->createExecutionContext();
    printf("Bindings after deserializing:\n");
    for (int bi = 0; bi < m_pEngine->getNbBindings(); bi++) 
	{
		Dims dimention = m_pEngine->getBindingDimensions(bi);
		std::cout << bi << " [" << m_maxBatchSize;
		for (int i = 0; i < dimention.nbDims; ++i) std::cout<< ", " << dimention.d[i];
		std::cout << "]. ";
        if (m_pEngine->bindingIsInput(bi) == true) printf("Binding (%s): Input.\n",   m_pEngine->getBindingName(bi));
        else printf("Binding (%s): Output.\n",  m_pEngine->getBindingName(bi));
    }
}

void copyDimention(const Dims& src, Dims& dst)
{
	dst.nbDims = src.nbDims;
	memcpy(dst.d, src.d, src.nbDims * sizeof(src.d[0]));
	for (int i = src.nbDims; i < 3; ++i) dst.d[i] = 1;
}

nvinfer1::DimsCHW TensorNet::allocateIOBuffer(size_t voBufferSize /*= 0*/)
{
	bool isDetetion = false;
	int nGPUBuffer = m_pEngine->getNbBindings();
	modelIObuffers.resize(nGPUBuffer);
	for (int b = 0; b < nGPUBuffer; b++) 
	{
		Dims dimention = m_pEngine->getBindingDimensions(b);
		size_t bites = std::accumulate(dimention.d, dimention.d + dimention.nbDims, sizeof(float), std::multiplies<int>());
		if ((b+1) == nGPUBuffer && bites < voBufferSize) bites = voBufferSize;
		bites *= m_maxBatchSize;
		cudaMalloc(&modelIObuffers[b], bites);
		if (0 == b)
		{
			modelInputBytes = bites/m_maxBatchSize;
			copyDimention(dimention, m_indims);
		}
		 else if ((b + 1) == nGPUBuffer)
		 {
			 copyDimention(dimention, m_dimsOut);
			 modelOutputBytes = bites/m_maxBatchSize;
			 std::string outBlobName(m_pEngine->getBindingName(b));
			 int index = outBlobName.find("detection");
			 isDetetion = (index >= 0 && index < outBlobName.size());
			 resultBufferHost = allocateLockedCPUMemory(bites);
			 std::cout << "model output shape: " << bites << ". (" << m_maxBatchSize << "," << m_dimsOut.c() << "," << m_dimsOut.h() << "," << m_dimsOut.w() << ")." << std::endl;
		 }
	}

	if (isDetetion)
    {
        imagePreprocessor = new CImagePreprocessor;
        imagePreprocessor->setOutput((float*)modelIObuffers[0]);
        imagePreprocessor->setSizeResizedImage(m_indims.w(), m_indims.h(), m_maxBatchSize);
    }
    return m_dimsOut;
}

//#include "caffePlugin.h"
#include "caculate3Dkeypoints.h"
void TensorNet::imageInference(int vBatchSize, const std::string& vName/*=""*/)
{
    if (NULL == m_pContext)
    {
        m_pContext = m_pEngine->createExecutionContext();
// 		m_pContext->setProfiler(&m_profiler);
        cudaEventCreate(&startCuda);
        cudaEventCreate(&endCuda);
//        m_pluginFactory.destroyPlugin();
        m_pInfer->destroy();
    }
    //CCaffePlugin::_printGPUdata((float*)modelIObuffers[0], 512 * 288 * 3, cudaStream, "ssd src");
    if (5 == /*++*/m_iter % 50)cudaEventRecord(startCuda, cudaStream);
    m_pContext->enqueue(vBatchSize, modelIObuffers.data(), cudaStream, NULL);
    if (5 == m_iter % 50)cudaEventRecord(endCuda, cudaStream);
// 	m_pContext->execute(batchSize, modelIObuffers);
    if (5 == m_iter % 50)cudaEventSynchronize(endCuda);
    if (5 == m_iter % 50)
    {
        float totalTime = 0;
        cudaEventElapsedTime(&totalTime, startCuda, endCuda);
        std::cout << m_iter << ", cudaInferenceTime = " << totalTime << std::endl;
    }

    if (vName != "keyPointInference" && vName != "ssdInference")
        cudaMemcpyAsync(resultBufferHost, modelIObuffers.back(), modelOutputBytes*vBatchSize, cudaMemcpyDeviceToHost, cudaStream);
// 	cudaMemcpyAsync(resultBufferHost, modelIObuffers[3], modelOutputBytes, cudaMemcpyDeviceToHost, cudaStream);
}


int TensorNet::__getSSDdetections(float vConfidence)
{
// 	calculateSSDnum(vConfidence, (float*)modelIObuffers.back(), m_dimsOut.h(), cudaStream);
    cudaMemcpyAsync(resultBufferHost, modelIObuffers.back(), sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
    cudaStreamSynchronize(cudaStream);
    int numDetection = *resultBufferHost;
    cudaMemcpyAsync(resultBufferHost+1, (float*)modelIObuffers.back()+1, sizeof(float)*(numDetection*7-1), cudaMemcpyDeviceToHost, cudaStream);
    return numDetection;
}

DimsCHW TensorNet::getTensorDims(const char* name)
{
    for (int b = 0; b < m_pEngine->getNbBindings(); b++) {
        if( !strcmp( name, m_pEngine->getBindingName(b)) )
            return static_cast<DimsCHW&&>(m_pEngine->getBindingDimensions(b));
    }
    return DimsCHW{0,0,0};
}


void TensorNet::printTimes(int iteration)
{
    m_profiler.printLayerTimes(iteration);
}

void TensorNet::destroy()
{
//    m_pluginFactory.destroyPlugin();
    m_pEngine->destroy();
    m_pInfer->destroy();
    m_pContext->destroy();
}

#include "Cuda3DArray.h"
#include "postPara3DKeyPoints.pb.h"
#include "protoIO.h"
int TensorNet::caculateDetail3Dkeypoints(int vBatchSize, int* vpBoxes)
{
    auto pSrc = vpBoxes;
    float *pDst = (float *)vpBoxes;
    int n = vBatchSize * 4;
    while(n--)
        *pDst++ = *pSrc++;
    pDst = (float *)vpBoxes;
    float* dDst = (float*)modelIObuffers.back() + modelOutputBytes*m_maxBatchSize/4 - vBatchSize  * 4;

    cudaMemcpyAsync(dDst, pDst, vBatchSize * 4 * 4, cudaMemcpyHostToDevice, cudaStream);

    if (m_pshpBase == NULL)
    {
        cudaChannelFormatDesc floatElementFormat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        m_pshpBase = new CCuda3DArray(40, 204, 0, floatElementFormat);
        m_pexpBase =  new CCuda3DArray(10, 204, 0, floatElementFormat);
        C3DPara para;
        CProtoIO pio;
        pio.readProtoFromBinaryFile("/srv/models/postPara3DKeyPoints.dat", &para);
		std::cout << "3d post " << para.wshpbase_size() << " : " << para.wshpbase().data()[0] << std::endl;
		m_pshpBase->copyData2GPUArray(para.wshpbase().data(), -1, cudaStream);
        m_pexpBase->copyData2GPUArray(para.wexpbase().data(), -1, cudaStream);
		std::cout << para.wexpbase_size() << " : " << para.wexpbase(0) << std::endl;
        exitIfCudaError(cudaStreamSynchronize(cudaStream));
        m_shpBase = m_pshpBase->offerTextureObject(false, cudaFilterModePoint, cudaReadModeElementType);
        m_expBase = m_pexpBase->offerTextureObject(false, cudaFilterModePoint, cudaReadModeElementType);
        para.Clear();
    }
    caculate3Dkeypoints((float*)modelIObuffers.back(), vBatchSize, dDst, m_shpBase, m_expBase, cudaStream);
    cudaMemcpyAsync(resultBufferHost, modelIObuffers.back(), modelOutputBytes*vBatchSize, cudaMemcpyDeviceToHost, cudaStream);
//    cudaMemcpyAsync(resultBufferHost, dDst, 219*vBatchSize, cudaMemcpyDeviceToHost, cudaStream);
    return modelOutputBytes*vBatchSize;
}

float getLengthOfIntersect(float a0, float a1, float b0, float b1)
{
    float right = std::min(a1, b1);
    float left = std::max(a0, b0);
    return  right - left;
}
float getArea(const float* vp)
{
    float x = vp[2] - vp[0];
    float y = vp[3] - vp[1];
    return x*y;
}
float caculateIoU(const float* vpA, const float* vpB)
{
    float iX = getLengthOfIntersect(vpA[0], vpA[2], vpB[0], vpB[2]);
    if (iX <= 0) return 0;
    float iY = getLengthOfIntersect(vpA[1], vpA[3], vpB[1], vpB[3]);
    if (iY <= 0) return 0;
    float i = iX * iY;
    float u0 = getArea(vpA);
    float u1 = getArea(vpB);
    float u = u0 + u1 - i;
    return i / u;
}
#include <list>
float caculateMaxIoUandErase(const float* vpA, std::list<float*>& vpBs, float vIouThresh, std::vector<float*>& vPaired)
{
    float maxIoU = 0;
    auto temp = vpBs.cend();
    for (auto p = vpBs.cbegin(); p != vpBs.cend(); ++p)
    {
        float iou = caculateIoU(vpA+3, *p+3);
        if (iou > vIouThresh && maxIoU < iou)
        {
            temp = p; maxIoU = iou;
        }
    }
	if (maxIoU > vIouThresh)
	{
		vPaired.emplace_back(*temp);
		vpBs.erase(temp);
	}
    return maxIoU;
}

#include "postDetections.h"

void __getheadFacePair(float* viopDetections, std::vector<float>& vbuffer)
{
    int headId = 1, faceId = 2 ;
    int numDetection = *viopDetections+0.2;
    float* pFace = viopDetections;
    SFloat7 f7; f7.f[1] = headId;
    SFloat7* pf7 = (SFloat7*)viopDetections;
    float *pHead = (float *)std::lower_bound(pf7, pf7 + numDetection, f7, [](const SFloat7& vA, const SFloat7& vB) {return vA.f[1] > vB.f[1]; });
    float* pLast = viopDetections + numDetection * 7;
    int numFace = (pHead - viopDetections) / 7;
    int numHead = numDetection - numFace;

    float* p = pHead;
    std::list<float*> headSet;
    for (int k = 0; k < numHead; ++k)
    {
        headSet.push_back(p);
        p += 7;
    }

    bool flagDeleteFace = false;
    const float iouThresh = 0.03;
	std::vector<float*> paired;
    for (int i = 0; i < numFace; ++i)
    {
        float iou = caculateMaxIoUandErase(pFace, headSet, iouThresh, paired);
        if (iou <= iouThresh)
        {
            if (pFace[2] <= 0.9)
            {
                pFace[1] = -1;
                flagDeleteFace = true;
            }
            else
            {
                memcpy(pLast, pFace, 7 * sizeof(float));
 				paired.emplace_back(pLast);
				pLast[1] = headId;
				numDetection++;
                pLast += 7;
            }
        }
        pFace += 7;
    }

	int numElement = (paired.size() + headSet.size()) * 7;
	vbuffer.resize(numElement);
	p = vbuffer.data();
	for (auto pd : paired) { memcpy(p, pd, sizeof(float) * 7); p += 7; }
	for (auto pd : headSet) { memcpy(p, pd, sizeof(float) * 7); p += 7; }
	memcpy(pHead, vbuffer.data(), vbuffer.size() * sizeof(float));
	*viopDetections = numDetection;
// 	if (flagDeteteFace)
// 	{
// 		auto offset = std::find(pf7, pf7 + numFace, [](const SFloat7& vDetection) {return vDetection.f[0] < 0; });
// 		auto last =  std::remove_if(offset, pf7 + numFace, [](const SFloat7& vDetection) {return vDetection.f[0]<0; });
// 		cudaMemcpyAsync()
// 	}

}
void TensorNet::postSSD(int vBatchSize, int vHeight, int vWidth)
{
    postDetections(vBatchSize, vHeight, vWidth, (float*)modelIObuffers.back(), m_dimsOut.h(), cudaStream);
    cudaMemcpyAsync(resultBufferHost, modelIObuffers.back(), modelOutputBytes*vBatchSize, cudaMemcpyDeviceToHost, cudaStream);

    cudaStreamSynchronize(cudaStream);
	for (int i=0; i<vBatchSize; ++i)
    __getheadFacePair(resultBufferHost+i*modelOutputBytes/sizeof(float), m_buffer);
}

// #include "affineInterface.h"
#include <time.h>

void TensorNet::affine(int vBatchSize, float *vp68points) {
    clock_t start, end;
    bool show_time = false;
    start = clock();
    affine_instance.stream = cudaStream;
    affine_instance.show_eclipse_time = show_time;
//    float *dev_img_out;
//    cudaMalloc(&dev_img_out, 112 * 112 * sizeof(float3) * 4);//  dev_img_out,
    affine_instance.affineInterface(vBatchSize, vp68points, imagePreprocessor->getSrcImg(),
                                    imagePreprocessor->getWidthSrcImg(),
                                    imagePreprocessor->getHeigtSrcImg(), (float *) modelIObuffers.front());
    end = clock();
    if (show_time)
        printf("Total execution time %3.4f ms\n", (double) (end - start) / CLOCKS_PER_SEC * 1000 / vBatchSize);
}

TensorNet::~TensorNet()
{
}
