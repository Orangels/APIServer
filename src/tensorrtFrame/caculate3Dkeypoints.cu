#include <cuda_runtime.h>
#include <stdio.h>
#include "constData.h"

__device__ float _caculateVector3Multiply(float* vA, float* vB)
{
	float fa_b = 0;
	float a = *vA, b = *vB;
	fa_b += a * b;
	vA++; vB++;
	a = *vA, b = *vB;
	fa_b += a * b;
	vA++; vB++;
	a = *vA, b = *vB;
	fa_b += a * b;
	return fa_b;
}

__device__ float _caculateVectorMultiply(cudaTextureObject_t vA, float* vB, int yTexture, int n)
{
	float fsum = 0;
	for (int i = 0; i < n; ++i)
	{
		float a = tex2D<float>(vA, i, yTexture);
		float b = vB[i];
		fsum += a * b;
	}
	return fsum;
}

__device__ void _caculateScale(const float* vipRect, float* voScale)
{
	float rightBottom = vipRect[2 + threadIdx.x];
	float leftUp = vipRect[threadIdx.x];
	voScale[threadIdx.x] = (rightBottom - leftUp) / 120;
	if (0 == threadIdx.x) voScale[2] = (voScale[0] + voScale[1]) / 2;
}

__device__ void _normalize(float* viop)
{
	float length = normf(3, viop+((threadIdx.x>>2)<<2));
	if (3 != (threadIdx.x & 3)) viop[threadIdx.x] /= length;
}
__device__ void _cross(float* A, float* B, float* C, int n)
{
	int i1 = (1 + threadIdx.x) % n;
	int i2 = (2 + threadIdx.x) % n;
	float c = A[i1] * B[i2];
	c -= A[i2] * B[i1];
	C[threadIdx.x] = c;
}

__device__ void _caculateEulerAngle(float* vpMat, float* vop)
{
	const float pi = 3.1415926f;
	float3 xyz;
	float R20 = vpMat[8];
	if (fabsf(R20 - 1) > 1e-4f || fabsf(R20 + 1) > 1e-4f)
	{
		xyz.x = asin(R20);
		float cosX = cos(xyz.x);
		xyz.y = atan2(vpMat[9] / cosX, vpMat[10] / cosX);
		xyz.z = atan2(vpMat[4] / cosX, vpMat[0] / cosX);
	}
	else
	{
		xyz.z = 0;
		if (fabsf(R20 + 1) < 1e-4f)
		{
			xyz.x = pi / 2;
			xyz.y = atan2(vpMat[1], vpMat[2]);
		}
		else
		{
			xyz.x = -pi / 2;
			xyz.y = atan2(-vpMat[1], -vpMat[2]);
		}
	}
	xyz.x *= 180.f / pi;
	xyz.y *= 180.f / pi;
	xyz.z *= 180.f / pi;
	*(float3*)vop = xyz;
}
__global__ void get3Dkerel(float* viopKeyPoints, float* vpBoxes, cudaTextureObject_t w_shp_base, cudaTextureObject_t w_exp_base)
{
	__shared__ float s_src[62];
	int tid = threadIdx.x;
	if (threadIdx.x < 62)
	{
		float src = viopKeyPoints[62 * blockIdx.x+ threadIdx.x];
		src *= param_std[threadIdx.x];
		src += param_mean[threadIdx.x];
		s_src[threadIdx.x] = src;
	}
	__syncthreads();
	__shared__ float vertex[204];
	float temp = u_base[threadIdx.x];
	temp +=_caculateVectorMultiply(w_shp_base, s_src + 12, threadIdx.x, 40);
	temp += _caculateVectorMultiply(w_exp_base, s_src + 52, threadIdx.x, 10);
	vertex[threadIdx.x] = temp;

	__syncthreads();
	int xyz = threadIdx.x / 68;
	temp = _caculateVector3Multiply(s_src + (xyz << 2), vertex + (threadIdx.x %68)* 3);
	temp += s_src[(xyz << 2) + 3]; 
	if (1 == xyz) temp = 121 - temp;
	__shared__ float s_scale[3];
	float* pBoxes = vpBoxes + (blockIdx.x << 2);
	if (threadIdx.x < 2) _caculateScale(pBoxes, s_scale);

	__syncthreads();
	temp *= s_scale[xyz];
	if (2 != xyz) temp += pBoxes[xyz];
	int offset = gridDim.x * 62 + blockIdx.x*219 +  threadIdx.x;
	viopKeyPoints[offset] = temp;

	if (threadIdx.x < 8)_normalize(s_src);
	if (threadIdx.x < 3) _cross(s_src, s_src + 4, s_src + 8, 3);
	if (31 == threadIdx.x)_caculateEulerAngle(s_src, s_src + 12);
	if (threadIdx.x < 12)
	{
		offset +=  204;
		viopKeyPoints[offset] = s_src[threadIdx.x];
	}
	if (threadIdx.x < 3)
	{
		offset +=  12;
		viopKeyPoints[offset] = s_src[12+threadIdx.x];
	}
	//*/
}

void caculate3Dkeypoints(float* viopKeyPoints, int vBatchSize, float* vpBoxes, cudaTextureObject_t vShpBase, cudaTextureObject_t vExpBase, cudaStream_t vCudaStream)
{
	get3Dkerel << <vBatchSize, 204, 0, vCudaStream >> > (viopKeyPoints, vpBoxes, vShpBase,vExpBase);
}