#include "ImagePreprocessor.h"
#include <opencv2/opencv.hpp>
#include "Cuda3DArray.h"
//__constant__ SKernelPara vPara[1];

__global__ void preprocessKernel(SParaDynamic vPara)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	float4 PixelBGRA;
	if (tid < vPara.NumPixelResized)
	{
		float x = tid % vPara.DstWidth;
		float y = tid / vPara.DstWidth;
		x = (x + 0.5f) * vPara.xScaleS_R;
		y = (y + 0.5f) * vPara.yScaleS_R;
		PixelBGRA = tex2D<float4>(vPara.SrcImage, x, y);
		PixelBGRA.x *= 255.f;
		PixelBGRA.y *= 255.f;
		PixelBGRA.z *= 255.f;

//		PixelBGRA.x -= vPara.MeanValue[0];
//		PixelBGRA.y -= vPara.MeanValue[1];
//		PixelBGRA.z -= vPara.MeanValue[2];
	}

	__syncthreads();
	if (tid < vPara.NumPixelResized)
	{
		float* pOut = vPara.dpInputBlob + tid;
		*pOut = PixelBGRA.x; pOut += vPara.NumPixelResized;
		*pOut = PixelBGRA.y; pOut += vPara.NumPixelResized;
		*pOut = PixelBGRA.z;
	}
}

#define readPixelM(Pixel, X, Y) {Pixel.x = tex2D<uchar>(SrcImage, X, Y); Pixel.y = tex2D<uchar>(SrcImage, X+1.f, Y); Pixel.z = tex2D<uchar>(SrcImage, X+2.f, Y);}

__device__ void appandColor(float3& voRBG, uchar3 vPixel, float xRelativity, float yRelativity)
{
	float Relativity = xRelativity * yRelativity;
	voRBG.x += vPixel.x * Relativity;
	voRBG.y += vPixel.y * Relativity;
	voRBG.z += vPixel.z * Relativity;
}

__global__ void preprocessKernel2(SParaDynamic vPara, int vXtopLeft=0, int vYtopLeft=0)
{
	//if (0 == threadIdx.x) printf("dddd%d\n", blockIdx.x);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int NumPixelResized = vPara.NumPixelResized;
	if (tid < NumPixelResized)
	{
		//����ԭ���� �� ����
		float X = tid % vPara.DstWidth;
		float Y = tid / vPara.DstWidth;
		X = (X + 0.5f) * vPara.xScaleS_R - 0.5f;
		Y = (Y + 0.5f) * vPara.yScaleS_R - 0.5f;
		int x = X, y = Y;
		float xDistance = X - x;
		float yDistance = Y - y;
		x += x << 1;
		X = x + 0.5f;
		Y = y + 0.5f;
		X += vXtopLeft;
		Y += vYtopLeft;

		//���ĸ���
		uchar3 TLPixel, TRPixel, BLPixel, BRPixel;
		cudaTextureObject_t SrcImage = vPara.SrcImage;
		__syncthreads();
		readPixelM(TLPixel, X, Y);
		readPixelM(TRPixel, X + 3.f, Y);
		readPixelM(BLPixel, X, Y + 1.f);
		readPixelM(BRPixel, X + 3.f, Y + 1.f);

		//����
		float3 RBG{ 0 };
		//memcpy(&RBG, vPara.MeanValue, sizeof(RBG));
		appandColor(RBG, TLPixel, 1.f - xDistance, 1.f - yDistance);
		appandColor(RBG, TRPixel, xDistance, 1.f - yDistance);
		appandColor(RBG, BLPixel, 1.f - xDistance, yDistance);
		appandColor(RBG, BRPixel, xDistance, yDistance);

		float* pOut = vPara.dpInputBlob + tid;
		__syncthreads();//д��
		*pOut = RBG.x; pOut += NumPixelResized;
		*pOut = RBG.y; pOut += NumPixelResized;
		*pOut = RBG.z;
	}
}

void CImagePreprocessor::preprocess(const cv::Mat& vSrcImage, cudaStream_t vStream)
{
	if (vSrcImage.rows < 10 || vSrcImage.rows>10000 || vSrcImage.cols < 10 || vSrcImage.cols>10000)
	{
		std::cout << "Invalid input image " << vSrcImage.cols << "x" << vSrcImage.rows << std::endl;
		return;
	}
	if (1 == m_iter)
	{
		cudaEventCreate(&startCopy);
		cudaEventCreate(&endCopy);
		cudaEventCreate(&endResize);
	}

	__checkIfSizeChange(vSrcImage);
	if (m_srcImageSizeChanged)
	{
		if (NULL == m_resizePraraD.dpInputBlob) __MallocGPUMemory();
		//if (NULL == m_cuda3DArray) exitIfCudaError(cudaMemcpyToSymbolAsync(vPara, &m_resizePraraD, sizeof(SKernelPara), 0, cudaMemcpyHostToDevice, vStream));
		m_resizePraraD.xScaleS_R = m_srcWidth  * 1.0 / m_resizePraraD.DstWidth;
		m_resizePraraD.yScaleS_R = m_srcHeight * 1.0 / m_resizePraraD.DstHeight;
		cudaChannelFormatDesc ArrayElementFormat = cudaCreateChannelDesc(8, 8 * (m_srcChanle == 4), 8 * (m_srcChanle == 4), 8 * (m_srcChanle == 4), cudaChannelFormatKindUnsigned);
		m_cuda3DArray = new CCuda3DArray(m_srcWidth*(m_srcChanle == 3 ? 3 : 1), m_srcHeight, 0, ArrayElementFormat, false);
		if (4 == m_srcChanle) m_resizePraraD.SrcImage = m_cuda3DArray->offerTextureObject(false, cudaFilterModeLinear, cudaReadModeNormalizedFloat);
		else m_resizePraraD.SrcImage = m_cuda3DArray->offerTextureObject(false, cudaFilterModePoint, cudaReadModeElementType);
		m_cropRePara.SrcImage = m_resizePraraD.SrcImage;
	}

	const int ThreadPerBlock = 512;
	float coppTime = 0, resizeTime=0;
	if (5 == m_iter % 50)cudaEventRecord(startCopy, vStream);
	m_cuda3DArray->copyData2GPUArray(m_resizeBuffer->data, -1, vStream);
	if (5 == m_iter % 50)
	{
		cudaEventRecord(endCopy, vStream);
		cudaEventSynchronize(endCopy);
		cudaEventElapsedTime(&coppTime, startCopy, endCopy);
	}
	const int NumBlock = (m_resizePraraD.NumPixelResized + ThreadPerBlock - 1) / ThreadPerBlock;
	if (5 == m_iter % 50)cudaEventRecord(startResize, vStream);
	if (4 == m_srcChanle) preprocessKernel << <NumBlock, ThreadPerBlock, 0, vStream >> >(m_resizePraraD);
	else if (3 == m_srcChanle) preprocessKernel2 << <NumBlock, ThreadPerBlock, 0, vStream >> >(m_resizePraraD);
	if (5 == m_iter % 50)cudaEventRecord(endResize, vStream);
	if (5 == m_iter % 50)
	{
		static float sumResize = 0;
		cudaEventSynchronize(endResize);
		cudaEventElapsedTime(&resizeTime, startResize, endResize);
		sumResize += resizeTime;
		std::cout << m_iter << ", coppTime, resizeTime, sumResize = " << coppTime <<", " <<  resizeTime << ", " << sumResize << std::endl;
	}
	//m_iter++;
}


void CImagePreprocessor::preprocess(const std::vector<cv::Mat>& vSrcImageSet, cudaStream_t vStream)
{
	float* dp = m_resizePraraD.dpInputBlob;
	for (int i = 0; i < vSrcImageSet.size(); ++i)
	{
		preprocess(vSrcImageSet[i], vStream);
		if (i < vSrcImageSet.size() - 1) 
			exitIfCudaError(cudaStreamSynchronize(vStream));
		m_resizePraraD.dpInputBlob += m_resizePraraD.NumPixelResized * 3;
	}
	m_resizePraraD.dpInputBlob = dp;
}

__global__ void readImage(float* vioImage, int vNumPixel)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	uchar3 bgr;
	float* src = vioImage + tid;
	bgr.x = (uchar)*src;
	bgr.y = (uchar)*(src+vNumPixel);
	bgr.z = (uchar)*(src+(vNumPixel<<1));
	memcpy(((uchar*)vioImage)+tid+(tid<<1), &bgr.x, 3);
}
void CImagePreprocessor::getResizedImage(float* srcGPU, char* dstCPU, int vWidth, int vHeight, cudaStream_t vStream)
{
	int numPixel = vHeight * vWidth;
	readImage << <vHeight, vWidth, 0, vStream >> > (srcGPU, numPixel);
	cudaMemcpyAsync(dstCPU, srcGPU, numPixel*3, cudaMemcpyDeviceToHost, vStream);
}


void CImagePreprocessor::__showGPUImage(int vWidth, int vHeight, float* vpImageGPU, cudaStream_t vStream)
{
	cv::Mat imgCroped(vHeight, vWidth, CV_8UC3);
	float* pHost = new float[vWidth * vHeight * 3];
	float *p1 = pHost + vWidth * vHeight, *p2 = p1 + vWidth * vHeight;
	cudaMemcpyAsync(pHost, vpImageGPU, vWidth * vHeight * 3 * 4, cudaMemcpyDeviceToHost, vStream);
	cudaStreamSynchronize(vStream);
	uchar *p = (uchar*)imgCroped.data;
	float *p00 = pHost, *p11 = p1, *p22 = p2;
	for (int i = 0; i < 120 * 120; ++i)
	{
		*p = *p00; p++; p00++;
		*p = *p11; p++; p11++;
		*p = *p22; p++; p22++;
	}
	cv::imshow("croped", imgCroped);
	cv::waitKey(0);
}


void CImagePreprocessor::cropResize(int vBatch, int* vpRects, cudaStream_t vStream)
{
	if (vBatch > m_cropCudaStream.size())
	{
		std::cout << "batchSize error: " << vBatch << " > " << m_cropCudaStream.size() << std::endl;
		exitIfCudaError(cudaErrorLaunchOutOfResources);
	}
	m_cropCudaStream.back() = vStream;
	const int ThreadPerBlock = 512;	
	const int NumBlock = (m_cropRePara.NumPixelResized + ThreadPerBlock - 1) / ThreadPerBlock;
	int s = m_cropCudaStream.size() - vBatch;
	float *src = m_cropRePara.dpInputBlob;
	for (int i = 0; i < vBatch; ++i)
	{	
		if (__invalidCrop(vpRects))
		{
			std::cout << "invalide crop rect: " << *(vpRects + 0) << ", " << *(vpRects + 1) << ", " << *(vpRects + 2) << ", " << *(vpRects + 3) << std::endl;
			exitIfCudaError(cudaErrorInvalidValue);
		}

		float width = *(vpRects + 2) - *(vpRects + 0) + 1;
		float heght = *(vpRects + 3) - *(vpRects + 1) + 1;
		m_cropRePara.xScaleS_R = width / m_cropRePara.DstWidth;
		m_cropRePara.yScaleS_R = heght / m_cropRePara.DstHeight;
		preprocessKernel2 << <NumBlock, ThreadPerBlock, 0, m_cropCudaStream[s+i] >> >(m_cropRePara, *vpRects*3, *(vpRects+1));
		//__showGPUImage(m_cropRePara.DstWidth, m_cropRePara.DstHeight, m_cropRePara.dpInputBlob, vStream);
		m_cropRePara.dpInputBlob += m_cropRePara.NumPixelResized * 3;
		vpRects += 4;
	}
	m_cropRePara.dpInputBlob = src;

	for (int i = 0; i < (vBatch - 1); ++i)
	{
		cudaStreamSynchronize(m_cropCudaStream[s + i]);
	}
}
#include<algorithm>
bool CImagePreprocessor::__checkIfSizeChange(const cv::Mat& vNewImage)
{
	//*m_resizeBuffer = (cv::Mat&)vNewImage;
	m_widthRatio = __iPowerDown(vNewImage.cols / m_resizePraraD.DstWidth);
	m_heghtRatio = __iPowerDown(vNewImage.rows / m_resizePraraD.DstHeight);
	int ratio = m_widthRatio < m_heghtRatio ? m_widthRatio : m_heghtRatio;
	if (ratio > 1)  m_widthRatio = m_heghtRatio = ratio =1 ;
	size_t captal = m_resizeBuffer->cols* m_resizeBuffer->rows * m_resizeBuffer->channels();
	size_t numEle = (vNewImage.cols/ m_widthRatio)* (vNewImage.rows/ m_widthRatio) * vNewImage.channels();
	if (captal < numEle)
	{
		if (m_cpuBuffer != NULL)cudaFreeHost(m_cpuBuffer);
		cudaMallocHost((void**)&m_cpuBuffer, numEle);
		delete m_resizeBuffer;
		//m_resizeBuffer = new cv::Mat(vNewImage.rows / m_heghtRatio, vNewImage.cols / m_widthRatio, CV_8UC3, m_cpuBuffer);
	}
	
	m_srcImageSizeChanged = !((m_srcWidth == m_resizeBuffer->cols) && (m_srcHeight == m_resizeBuffer->rows) && (m_srcChanle == m_resizeBuffer->channels()));
	if (m_srcImageSizeChanged) 
	{ 	m_resizeBuffer = new cv::Mat(vNewImage.rows / m_heghtRatio, vNewImage.cols / m_widthRatio, CV_8UC3, m_cpuBuffer);
		m_srcWidth = m_resizeBuffer->cols; m_srcHeight = m_resizeBuffer->rows; m_srcChanle = m_resizeBuffer->channels();		
	}
	if (m_widthRatio == 1 && m_heghtRatio == 1)
		memcpy(m_cpuBuffer, vNewImage.data, vNewImage.rows*vNewImage.cols*vNewImage.channels());
	else
		cv::resize(vNewImage, *m_resizeBuffer, cv::Size(m_resizeBuffer->cols, m_resizeBuffer->rows));
	return m_srcImageSizeChanged;
}

void CImagePreprocessor::__MallocGPUMemory()
{
	if (NULL == m_resizePraraD.dpInputBlob)
	{
		size_t BufferSize = m_resizePraraD.NumPixelResized * 3 * sizeof(float);
		exitIfCudaError(cudaMalloc((void**)&m_resizePraraD.dpInputBlob, BufferSize));
		m_IsMallocInteral = true;
	}
}


void CImagePreprocessor::setSizeResizedImage(int vWidth, int vHeight, int vMaxBatch, bool vCrop)
{
	if (!vCrop)
	{
		m_resizePraraD.DstWidth = vWidth;
		m_resizePraraD.DstHeight = vHeight;
		m_resizePraraD.NumPixelResized = vWidth * vHeight;
	}
	else
	{		
		m_cropRePara.DstWidth = vWidth;
		m_cropRePara.DstHeight = vHeight;
		m_cropRePara.NumPixelResized = vWidth * vHeight;
		m_cropCudaStream.resize(vMaxBatch);
		for (int i = 0; i < (vMaxBatch-1); ++i)
			cudaStreamCreate(&m_cropCudaStream[i]);
	}
}

void CImagePreprocessor::setMeanValue(const float* vMeanValue, int vBites /*= 12*/)
{
	//memcpy(m_resizePraraD.MeanValue, vMeanValue, vBites);
	//m_resizePraraD.MeanValue[0] *= -1;  m_resizePraraD.MeanValue[1] *= -1;  m_resizePraraD.MeanValue[2] *= -1;
}

void CImagePreprocessor::setOutput(float* voDeviceBufferForLearningInput /*= NULL*/, bool vCrop)
{
	if (!vCrop)
	m_resizePraraD.dpInputBlob = voDeviceBufferForLearningInput;
	else 
	m_cropRePara.dpInputBlob = voDeviceBufferForLearningInput;
}


cudaTextureObject_t CImagePreprocessor::getSrcImg()
{
return	m_cuda3DArray->offerTextureObject(false, cudaFilterModePoint, cudaReadModeElementType);
}

CImagePreprocessor::CImagePreprocessor() : m_cuda3DArray(NULL), m_srcImageSizeChanged(true), m_IsMallocInteral(false), m_srcHeight(0), m_srcWidth(0), m_srcChanle(0), m_resizeBuffer(new cv::Mat())
{
	m_resizePraraD.dpInputBlob = NULL;
}

CImagePreprocessor::~CImagePreprocessor()
{
	if (m_resizePraraD.dpInputBlob && m_IsMallocInteral)
	{
		cudaFree(m_resizePraraD.dpInputBlob);
		m_resizePraraD.dpInputBlob = NULL;
	}
	if (m_cuda3DArray)
	{
		delete m_cuda3DArray;
		m_cuda3DArray = NULL;
	}
	if (m_cpuBuffer != NULL)
	{
		cudaFreeHost(m_cpuBuffer);
		m_cpuBuffer = NULL;
		delete m_resizeBuffer;
	}
}


bool CImagePreprocessor::__invalidCrop(int* vpRect)
{
	const int bounder = 256;
//	std::cout <<"x1"<<*vpRect<<"y1"<<*(vpRect+1)<<"x2"<<*(vpRect+2)<<"y2"<<*(vpRect+3)<<std::endl;
//	std::cout <<"m_widthRatio"<<m_widthRatio<<std::endl;
//	std::cout <<"m_heghtRatio"<<m_heghtRatio<<std::endl;

	*vpRect /= float(m_widthRatio);

	int xMin = *vpRect; vpRect++;
	if (xMin < -bounder || xMin >= m_srcWidth)
		return true;
	*vpRect /= float(m_heghtRatio);
	int yMin = *vpRect; vpRect++;
	if (yMin < -bounder || yMin >= m_srcHeight)
		return true;
	*vpRect /= float(m_widthRatio);
	int xMax = *vpRect; vpRect++;
	if (xMax <= xMin - 1 || xMax >= m_srcWidth + bounder || xMax < 0)
		return true;
	*vpRect /= float(m_heghtRatio);
	int yMax = *vpRect; vpRect++;
	if (yMax <= yMin - 1 || yMax >= m_srcHeight + bounder || yMax < 0)
		return true;
	return false;
}
