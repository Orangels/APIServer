/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "cudaUtility.h"
#include <iostream>

/*
// gpuPreImageNet
__global__ void gpuPreImageNet( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z, px.y, px.x);
	
	output[n * 0 + y * oWidth + x] = bgr.x;
	output[n * 1 + y * oWidth + x] = bgr.y;
	output[n * 2 + y * oWidth + x] = bgr.z;
}

// cudaPreImageNet
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight,
				         float* output, size_t outputWidth, size_t outputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNet<<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}

// gpuPreImageNetMean
__global__ void gpuPreImageNetMean( float2 scale, float3* input, int iWidth, float* output, int oWidth, int oHeight, float3 mean_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float3 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z - mean_value.x, px.y - mean_value.y, px.x - mean_value.z);
	
	output[n * 0 + y * oWidth + x] = bgr.x;
	output[n * 1 + y * oWidth + x] = bgr.y;
	output[n * 2 + y * oWidth + x] = bgr.z;
}

// cudaPreImageNetMean
cudaError_t cudaPreImageNetMean( float3* input, size_t inputWidth, size_t inputHeight,
				             float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value )

{
	if( !input || !output ){
        std::cout << "error here. "<< std::endl;
        return cudaErrorInvalidDevicePointer;
    }

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 ){
        std::cout << "Or here. " << std::endl;
        return cudaErrorInvalidValue;
    }


	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );


	// launch kernel

	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreImageNetMean<<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight, mean_value);

	return CUDA(cudaGetLastError());

}

__global__ void kernel_extract_roi(float* input, float* output, char* mean,
    const int input_w, const int output_w, const int output_h,
    const int in_plane_r, const int in_plane_g, const int in_plane_b,
    const int out_plane_r, const int out_plane_g, const int out_plane_b,
    const int bbox_x, const int bbox_y, const int bbox_w, const int bbox_h)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if( x < output_w && y < output_h)
    {
        float r[2] = { float(x) * bbox_w / output_w + bbox_x,
                       float(y) * bbox_h / output_h + bbox_y };

        int   pos[4][2] = { { int(floor(r[0])), int(floor(r[1])) },
                            { int( ceil(r[0])), int(floor(r[1])) },
                            { int(floor(r[0])),  int(ceil(r[1])) },
                            { int( ceil(r[0])),  int(ceil(r[1])) } };

        float u = r[0]-floor(r[0]);
        float v = r[1]-floor(r[1]);

        float s[4] = { (1-u)*(1-v), u*(1-v), (1-u)*v, u*v };

        int map[4] = { pos[0][1]*input_w + pos[0][0], pos[1][1]*input_w + pos[1][0],
                       pos[2][1]*input_w + pos[2][0], pos[3][1]*input_w + pos[3][0]};

        int idx = y * output_w + x;
        output[idx+out_plane_r] = round( s[0]*input[map[0]+in_plane_r]
                                       + s[1]*input[map[1]+in_plane_r]
                                       + s[2]*input[map[2]+in_plane_r]
                                       + s[3]*input[map[3]+in_plane_r] );// float(mean[idx+out_plane_r]));
        output[idx+out_plane_g] = round( s[0]*input[map[0]+in_plane_g]
                                       + s[1]*input[map[1]+in_plane_g]
                                       + s[2]*input[map[2]+in_plane_g]
                                       + s[3]*input[map[3]+in_plane_g] );//float(mean[idx+out_plane_g]));
        output[idx+out_plane_b] = round( s[0]*input[map[0]+in_plane_b]
                                       + s[1]*input[map[1]+in_plane_b]
                                       + s[2]*input[map[2]+in_plane_b]
                                       + s[3]*input[map[3]+in_plane_b] );//float(mean[idx+out_plane_b]));
    }
}

void convertROI(float* input, float* output, char* mean, const int* srcSize, const int* dstSize, const int* roi, cudaStream_t stream)
{
    int in_plane_r = 0;
    int in_plane_g = srcSize[1] * srcSize[2];
    int in_plane_b = srcSize[1] * srcSize[2] * 2;

    int out_plane_r = 0;
    int out_plane_g = dstSize[1] * dstSize[2];
    int out_plane_b = dstSize[1] * dstSize[2] * 2;

    int bbox_x = min(max(roi[0], 0), srcSize[2]-1);
    int bbox_y = min(max(roi[1], 0), srcSize[1]-1);
    int bbox_w = min(max(roi[2]-roi[0], 0), srcSize[2]-bbox_x-1 );
    int bbox_h = min(max(roi[3]-roi[1], 0), srcSize[1]-bbox_y-1 );

    dim3 dimBlock(32,32);
    dim3 dimGrid(dstSize[2]/dimBlock.x+1, dstSize[1]/dimBlock.y+1);

    std::cout << "ROI: " << bbox_x << " " << bbox_y << " " << bbox_w << " " << bbox_h << std::endl;

    kernel_extract_roi <<< dimGrid, dimBlock, 0, stream >>> (input, output, mean,
                       srcSize[2], dstSize[2], dstSize[1],
                       in_plane_r,   in_plane_g,  in_plane_b,
                       out_plane_r, out_plane_g, out_plane_b,
                       bbox_x, bbox_y, bbox_w, bbox_h);
}

*/
__device__ float reductionAddShared(float* vpBuffer, int vN)
{// vN == power(2)
	__syncthreads();
	while (vN>>=1)
	{
		if (threadIdx.x < vN)
		{
			vpBuffer[threadIdx.x] += vpBuffer[threadIdx.x + vN];
		}
		__syncthreads();
	}
	return *vpBuffer;
}

__device__ float readAndExp(float* vSrc, int vOffset, int vBounder)
{
	float src = 0;
	if (vOffset < vBounder) { src = *(vSrc + vOffset); src = __expf(src);}
	return src;
}

__device__ void writeData(float* vpDst, float vData, int vOffset, int vBounder)
{
	if (vOffset < vBounder) *(vpDst + vOffset) = vData;
}

__global__  void kernelSoftmaxBiger32(float* x, int vL, int vLdowned, float* y, int vAll)
{
	extern __shared__ float s_buffer[];
	int tid = blockDim.x*blockIdx.x + threadIdx.x;
	float number = readAndExp(x, tid, vAll);
	float number2 = 0;
	if (threadIdx.x + vLdowned < vL) readAndExp(x, tid + vLdowned, vAll);
	s_buffer[threadIdx.x] = number + number2;

	float sum = reductionAddShared(s_buffer, vLdowned);

	writeData(y, __fdiv_rd(number, sum), tid, vAll);
	if (threadIdx.x + vLdowned < vL) writeData(y, __fdiv_rd(number2, sum), tid + vLdowned, vAll);
}

const int bounch = 32;
#define READUCEADD(a) 	__syncthreads(); if (L==a) { L >>=1; if(threadIdx.x<L){sum += s_bufferl[threadIdx.x + L][threadIdx.y]; s_bufferl[threadIdx.x][threadIdx.y] = sum;}}
__global__  void kernelSoftmaxLess32(float* x, int vL, int vLdowned, float* y, int vAll)
{
	extern __shared__ float s_bufferl[][bounch+1];
	int tid = blockDim.x * blockDim.y * blockIdx.x + blockDim.x* threadIdx.y + threadIdx.x;

	//read data
	float number = readAndExp(x, tid, vAll);
	if (threadIdx.x < vLdowned) s_bufferl[threadIdx.x][threadIdx.y] =  number;
	__syncthreads();
	if (threadIdx.x >= vLdowned) s_bufferl[threadIdx.x-vLdowned][threadIdx.y] += number;

	float sum = 0;	__syncthreads();
	if (threadIdx.x < vLdowned) sum = s_bufferl[threadIdx.x][threadIdx.y];
	int L = vLdowned;
	switch (L)
	{
		case 32:READUCEADD(32);
		case 16:READUCEADD(16);
		case 8:READUCEADD(8);
		case 4:READUCEADD(4);
		case 2:READUCEADD(2);
	}
	__syncthreads(); sum = s_bufferl[0][threadIdx.y];
	writeData(y, __fdiv_rd(number, sum), tid, vAll);
}


void cudaSoftmax(int nchw, int w,  float* x, float*y, cudaStream_t stream)
{
	int power2down = iPowerDown(w);
	if (w <= 32)
	{
		dim3 threads(w, bounch);
		int blocks = iDivUp(nchw, bounch * w);
		kernelSoftmaxLess32 <<<blocks, threads, (bounch+1) * power2down*sizeof(float), stream>>>(x, w, power2down, y, nchw);
	}
	else
	{
		int blocks = iDivUp(nchw, power2down);
		kernelSoftmaxBiger32 << <blocks, power2down, power2down * sizeof(float), stream >> >(x, w, power2down, y, nchw);
	}
}


__global__  void kernelPrelu(float* vpSrc, float* vpSlope, float* vpDst, int c,  int h)
{
	float slope = vpSlope[blockIdx.x%c];
	int id = blockDim.x * h *blockIdx.x + threadIdx.x;
	float *pSrc = vpSrc + id;
	float *pDst = vpDst + id;
	for (int i = 0; i < h; ++i)
	{
		float src = *pSrc;
		if (src < 0)
			src *= slope;
		*pDst = src;
		pSrc += blockDim.x;
		pDst += blockDim.x;
	}
}

void preluCuda(int n, int c, int h, int w, float* src, float* pSlope, float* dst, cudaStream_t stream)
{
	kernelPrelu << <n*c, w, 0, stream >> > (src, pSlope, dst, c, h);
}

extern __shared__ float s_src[];
__global__  void kernelNormalize(float* vpSrc, float* vpWeight, float* vpDst, int vC, int wh)
{
	int ic = (threadIdx.x >> 5);
	int nc = (blockDim.x >> 5);
	int batch = vC / nc;
	int offset = (blockIdx.x << 5) + wh * ic + (31&threadIdx.x);
	float* pSrc = vpSrc + offset;

	// calculate l2 length.
	__shared__ float s_sum[32];
	if (threadIdx.x < 32)
		s_sum[threadIdx.x] = 0;
	for (int i = 0; i < batch; ++i)
	{
		s_src[threadIdx.x] = *pSrc;
		__syncthreads();
		if (threadIdx.x < 32)
		{
			for (int k = 0; k < nc; ++k)
			{
				float src = s_src[threadIdx.x + (k << 5)];
				s_sum[threadIdx.x] += src*src;
			}
		}
		__syncthreads();
		pSrc += wh*nc;
	}

	if (threadIdx.x < 32)
	{
		float l = s_sum[threadIdx.x];
		l = sqrt(l+1e-10f);
		s_sum[threadIdx.x] = 1.0f / l;
	}

	// x = a*x/l
	if (threadIdx.x < vC)
		s_src[threadIdx.x] = vpWeight[threadIdx.x];

	__syncthreads();
	pSrc = vpSrc + offset;
	float* pDst = vpDst + offset;
	for (int i = 0; i < batch; ++i)
	{
		float src = *pSrc;
		float a = s_src[ic + i*nc];
		float l_ = s_sum[(threadIdx.x & 31)];
		src *= a * l_;
		*pDst = src;
		pSrc += wh*nc;
		pDst += wh*nc;
	}
}

void normalizeCuda(int n, int c, int h, int w, float* src, float* pWeight, float* dst, cudaStream_t stream)
{
	const int numThreadPerBlock = 512;
	int shareBites = sizeof(float) * (numThreadPerBlock);
	kernelNormalize << <w*h/32, numThreadPerBlock, shareBites, stream >> > (src, pWeight, dst, c, w*h);
}


__global__  void kernelInnerScale(float* vpSrc, float* vpSRC, float* vpDst, int c, int h)
{
	float src = vpSrc[blockIdx.x%c];
	int id = blockDim.x * h *blockIdx.x + threadIdx.x;
	float *p = vpSRC + id;
	float *pDst = vpDst + id;
	for (int i = 0; i < h; ++i)
	{
		float SRC = *p * src;
		*pDst = SRC;
		p += blockDim.x;
		pDst += blockDim.x;
	}
}

void innerScaleCuda(int n, int c, int h, int w, float* src, float* SRC, float* dst, cudaStream_t stream)
{
	kernelInnerScale << <n*c, w, 0, stream >> > (src, SRC, dst, c, h);
}

__global__  void kernelAvgChan(float* vpSrc,  float* vpDst, int c, int h)
{	 
	extern __shared__ double s_buf[];
	int id = blockDim.x * h *blockIdx.x + threadIdx.x;
	float *pSrc = vpSrc + id;
	double temp = 0;
	for (int i = 0; i < h; ++i)
	{
		temp += *pSrc;		
		pSrc += blockDim.x;
	}
	s_buf[threadIdx.x] = temp / h / blockDim.x;
	int n = blockDim.x;
	while (n>1)
	{
		if (n>32) __syncthreads();
		if (1 == (1 & n))
		{
			n--;
			if (threadIdx.x == 0)
				s_buf[0] += s_buf[n];
		}
		else
		{
			n >>= 1;
			if (threadIdx.x < n)
				s_buf[threadIdx.x] += s_buf[threadIdx.x + n];
		}
	}
	//__syncthreads();
	if (threadIdx.x == 0)
	{
		//printf("%.2f\n" , s_buf[0]);
		vpDst[blockIdx.x] = s_buf[0] ;
	}
}
void avgChanCuda(int n, int c, int h, int w, float* src, float* dst, cudaStream_t stream)
{
	kernelAvgChan << <n*c, w, w*sizeof(double), stream >> > (src,  dst, c, h);
}


__global__ void doSomeLoop(unsigned int *A, int vN)
{
	for (int i = 0; i < vN; ++i)
	{
		if (0 == threadIdx.x)
		{
			if (*A > 200000000)
				*A %= 11;
		}
		__syncthreads();
		atomicAdd(A, 3);
	}
}

void doCudaLoop(unsigned int* vp, int vn, cudaStream_t stream)
{
	doSomeLoop << <1, 2, 0, stream >> > (vp, vn);
}
