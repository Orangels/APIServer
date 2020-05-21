#include <cuda_runtime.h>
#include "postDetections.h"
#include "Common.h"
#include "Cuda3DArray.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

__constant__ float g_filterPara[5] = { 0.5f, 4096.f, 2.f };
__global__ void filtetrBoxesAndSort(float* viopDetections, cudaTextureObject_t vPara, int vHeight, int vWidth)
{//img_area=800000, min_thresh=0.5, min_abso_area=50 * 50, min_rela_area=0.01, max_aspect_ratio = 2.0
	float* pio = viopDetections + blockDim.x * 7 * blockIdx.x;

	float temp = g_filterPara[0];// tex2D<float>(vPara, 0, 0);//confidenceThresh
	__shared__ float sConsIndex[100];
	float conf = pio[2 + threadIdx.x * 7];
	sConsIndex[threadIdx.x] = conf;
	__syncthreads();
	__shared__ int sNumCondidat;
	if (conf >= temp || 0 == threadIdx.x)
	{
		if (sConsIndex[threadIdx.x + 1] < temp)
			sNumCondidat = threadIdx.x + (conf >= temp);
	}
	__syncthreads();
	int numCondidat = sNumCondidat;
	
	//area max_aspect
	const float exclude = -1;
	if (threadIdx.x < numCondidat)
	{
		float4 xyMinMax;// = *(float4*)(pio + 3 + threadIdx.x * 7);
		memcpy(&xyMinMax, pio + 3 + threadIdx.x * 7, sizeof(xyMinMax));
		xyMinMax.x *= vWidth; xyMinMax.z *= vWidth;
		xyMinMax.y *= vHeight; xyMinMax.w *= vHeight;
		//*(float4*)(pio + 3 + threadIdx.x * 7) = xyMinMax;
		memcpy(pio + 3 + threadIdx.x * 7, &xyMinMax, sizeof(xyMinMax));
		xyMinMax.z -= xyMinMax.x;//width
		xyMinMax.w -= xyMinMax.y;//hegth
		xyMinMax.x = xyMinMax.z*xyMinMax.w;//area
		temp = g_filterPara[1];// tex2D<float>(vPara, 1, 0);//areaThresh  
		if (xyMinMax.x < temp) sConsIndex[threadIdx.x] = exclude;
		else
		{
			temp = g_filterPara[2];//tex2D<float>(vPara, 2, 0);//max_aspectThresh 
			xyMinMax.y = xyMinMax.z / xyMinMax.w;//max_aspect
			if (1.f > xyMinMax.y) xyMinMax.y = 1.f / xyMinMax.y;
			if (xyMinMax.y > temp) sConsIndex[threadIdx.x] = exclude;
			else sConsIndex[threadIdx.x] = pio[1 + threadIdx.x * 7];
		}
	}
	__syncthreads();
	thrust::stable_sort_by_key(thrust::device, sConsIndex, sConsIndex + numCondidat, (SFloat7*)pio, [](const float& a, const float& b) {return a > b; });
	numCondidat = thrust::lower_bound(thrust::device, sConsIndex, sConsIndex + numCondidat, 0, [](const float& a, const float& b) {return a > b; }) - sConsIndex;
	//numCondidat -= thrust::count_if(thrust::device, sConsIndex, sConsIndex + numCondidat, [](float f) {return f > 5000.f; });
	__syncthreads();
	if (numCondidat > threadIdx.x)
	*(pio+7*threadIdx.x) = numCondidat*(threadIdx.x==0);
}

__constant__ float adaptR[][2] = {
{0.355067475559597, 0.119262390148535 },
{0.332986547478306, 0.246449822265788 },
{-0.331071700057979, -0.146350441272887},
{-0.145853845175441, -0.399418765180059} };
__global__ void faceRefine(float* viopDetections, int vIndexFace/*, int vHeight, int vWidth*/)
{
	float* pio = viopDetections + blockDim.x * 7 * blockIdx.x;
	int numBox = *pio;

	pio += 7 * threadIdx.x;
	if (threadIdx.x < numBox)
	{
		int classID = pio[1];
		if (vIndexFace == classID)
		{
			float4 xyMinMax;// = *(float4*)(pio + 3);
			memcpy(&xyMinMax, pio + 3, sizeof(xyMinMax));
// 			xyMinMax.x *= vWidth; xyMinMax.z *= vWidth;
// 			xyMinMax.y *= vHeight; xyMinMax.w *= vHeight;
			float2 center;
			center.x = (xyMinMax.x + xyMinMax.z)*0.5f;
			center.y = (xyMinMax.y + xyMinMax.w)*0.5f;
			xyMinMax.x -= center.x;
			xyMinMax.y -= center.y;
			xyMinMax.z -= center.x;
			xyMinMax.w -= center.y;

			float length = 4/(xyMinMax.z + xyMinMax.w - xyMinMax.x - xyMinMax.y);
			xyMinMax.x *= length; xyMinMax.y *= length; 
			xyMinMax.z *= length; xyMinMax.w *= length;
			float4 bbox_dt;
			float2 r = *(float2*)adaptR[0]; bbox_dt.x = r.x * xyMinMax.x + r.y* xyMinMax.y - r.x*xyMinMax.z - r.y*xyMinMax.w;
			r = *(float2*)adaptR[1]; bbox_dt.y = r.x * xyMinMax.x + r.y* xyMinMax.y - r.x*xyMinMax.z - r.y*xyMinMax.w;
			r = *(float2*)adaptR[2]; bbox_dt.z = r.x * xyMinMax.x + r.y* xyMinMax.y - r.x*xyMinMax.z - r.y*xyMinMax.w;
			r = *(float2*)adaptR[3]; bbox_dt.w = r.x * xyMinMax.x + r.y* xyMinMax.y - r.x*xyMinMax.z - r.y*xyMinMax.w;
			
			length = 1 / length;
			bbox_dt.x *= length; bbox_dt.y *= length;
			bbox_dt.z *= length; bbox_dt.w *= length;

			bbox_dt.x += center.x; bbox_dt.z += center.x;
			bbox_dt.y += center.y; bbox_dt.w += center.y;

			r.x = bbox_dt.z - bbox_dt.x;
			r.y = bbox_dt.w - bbox_dt.y;
			r.x *= r.x; r.y *= r.y;
			length = r.x + r.y;
			length = sqrtf(length);
			length = int(length*0.85f+0.5f);

			center.x = (bbox_dt.x + bbox_dt.z)*0.5f;
			center.y = (bbox_dt.y + bbox_dt.w)*0.5f;

			bbox_dt.x = center.x - 0.5f*length; if (bbox_dt.x < 0) bbox_dt.x = 0;
			bbox_dt.y = center.y - 0.43f*length; if (bbox_dt.y < 0) bbox_dt.y = 0;
			bbox_dt.z = bbox_dt.x + length; bbox_dt.w = bbox_dt.y + length;
// 			bbox_dt.x /= vWidth; bbox_dt.z /= vWidth;
// 			bbox_dt.y /= vHeight; bbox_dt.w /= vHeight;
// 			*(float4*)(pio + 3) = bbox_dt;
			memcpy(pio + 3, &bbox_dt, sizeof(bbox_dt));
		}
	}
}
#include <algorithm>
#include <functional>

void postDetections(int vBatchSize, int vHeight, int vWidth, float* viopDetections, int vkeep_top_k, cudaStream_t vStream)
{
	static cudaChannelFormatDesc cFmt = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	static CCuda3DArray para(3, 1, 0, cFmt);
	static bool first = true;
	if (first)
	{
		first = false;//min_thresh=0.5, min_area=0.01, max_aspect_ratio = 2.0
// 		float ps[] = { 0.5f, std::max(4096.f, 0.005f * vHeight*vWidth), 2.f };
// 		para.copyData2GPUArray(ps, -1, vStream);
// 		exitIfCudaError(cudaStreamSynchronize(vStream));
	}
	filtetrBoxesAndSort << <vBatchSize, vkeep_top_k, 0, vStream >> > (viopDetections, 0, vHeight, vWidth);
	faceRefine << <vBatchSize, vkeep_top_k/2+1, 0, vStream >> > (viopDetections, 2);
}
