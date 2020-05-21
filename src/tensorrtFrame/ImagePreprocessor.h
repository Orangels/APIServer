#pragma once
#include <vector>
#include <cuda_runtime.h>
// struct SKernelPara
// {
// 	int DstWidth, DstHeight, NumPixelResized;
// 	float /*MeanValue[3],*/ *dpInputBlob;
// };
struct SParaDynamic
{
	int DstWidth, DstHeight, NumPixelResized;
	float /*MeanValue[3],*/ *dpInputBlob;
	cudaTextureObject_t SrcImage;
	float xScaleS_R, yScaleS_R; //the size ratio between src image and resized image.
};

class CCuda3DArray;
namespace cv{ class Mat; }
class CImagePreprocessor
{
public:
	CImagePreprocessor();
	~CImagePreprocessor();
	void preprocess(const cv::Mat& vSrcImage, cudaStream_t vStream);//说明：如果vSrcImage的内存采用cudaMallocHost分配，性能最佳
	void cropResize(int vBatch, int* vpRects,  cudaStream_t vStream);//说明：如果vSrcImage的内存采用cudaMallocHost分配，性能最佳
	void getResizedImage(float* srcGPU, char* dstCPU, int vWidth, int vHeight, cudaStream_t vStream);
	void preprocess(const std::vector<cv::Mat>& vSrcImageSet, cudaStream_t vStream);

	void setSizeResizedImage(int vWidth, int vHeight, int vMaxBatch=1, bool vCrop=false);
	inline int getDstWidth(){ return m_resizePraraD.DstWidth; }
	inline int getDstHeight(){ return m_resizePraraD.DstHeight; }
	void setMeanValue(const float* vMeanValue, int vBites = 12);
	void setOutput(float* voDeviceBufferForLearningInput = NULL, bool vCrop=false);
	inline float* getOutput(){ __MallocGPUMemory(); return m_resizePraraD.dpInputBlob; }

	cudaTextureObject_t getSrcImg();
	int getWidthSrcImg() { return m_srcWidth; }
	int getHeigtSrcImg() { return m_srcHeight; }
private:
	long long m_iter = 0;
	cudaEvent_t startCopy, endCopy, startResize, endResize;
	inline int __iPowerDown(int a) { int b = 1; while (a >>= 1) b <<= 1; return b; }
	bool __checkIfSizeChange(const cv::Mat& vNewImage);
	int m_srcWidth, m_srcHeight, m_srcChanle;
	CCuda3DArray*  m_cuda3DArray;
	bool m_srcImageSizeChanged;
	//SKernelPara m_KernelPara;
	void __MallocGPUMemory();
	bool m_IsMallocInteral;
	SParaDynamic m_resizePraraD, m_cropRePara;
	std::vector<cudaStream_t> m_cropCudaStream;
	void __showGPUImage(int vWidth, int vHeight, float* vpImageGPU, cudaStream_t vStream);
	bool __invalidCrop(int* vpRect);
	cv::Mat* m_resizeBuffer;
	unsigned char* m_cpuBuffer = NULL;
	int m_widthRatio, m_heghtRatio;
};


