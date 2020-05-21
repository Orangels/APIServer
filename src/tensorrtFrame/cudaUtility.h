/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __CUDA_UTILITY_H_
#define __CUDA_UTILITY_H_


#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>


/**
 * Execute a CUDA call and print out any errors
 * @return the original cudaError_t result
 * @ingroup util
 */
#define CUDA(x)				cudaCheckError((x), #x, __FILE__, __LINE__)

/**
 * Evaluates to true on success
 * @ingroup util
 */
#define CUDA_SUCCESS(x)			(CUDA(x) == cudaSuccess)

/**
 * Evaluates to true on failure
 * @ingroup util
 */
#define CUDA_FAILED(x)			(CUDA(x) != cudaSuccess)

/**
 * Return from the boolean function if CUDA call fails
 * @ingroup util
 */
#define CUDA_VERIFY(x)			if(CUDA_FAILED(x))	return false;

/**
 * LOG_CUDA string.
 * @ingroup util
 */
#define LOG_CUDA "[cuda]   "

/*
 * define this if you want all cuda calls to be printed
 */
//#define CUDA_TRACE



/**
 * cudaCheckError
 * @ingroup util
 */
inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt, const char* file, int line )
{
#if !defined(CUDA_TRACE)
	if( retval == cudaSuccess)
		return cudaSuccess;
#endif

	//int activeDevice = -1;
	//cudaGetDevice(&activeDevice);

	//Log("[cuda]   device %i  -  %s\n", activeDevice, txt);
	
	printf(LOG_CUDA "%s\n", txt);


	if( retval != cudaSuccess )
	{
		printf(LOG_CUDA "   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval), retval, retval);
		printf(LOG_CUDA "   %s:%i\n", file, line);	
	}

	return retval;
}

inline __device__ __host__ int iDivUp(int a, int b) { return (a + b - 1) / b; }
inline __device__ __host__ int iPowerDown(int a) { int b = 1; while (a >>= 1) b <<= 1; return b; }
inline __device__ __host__ int iPowerUp(int a) { int b = 1; while (b<a) b <<= 1; return b; }
void cudaSoftmax(int nchw, int w, float* x, float*y, cudaStream_t stream);
void preluCuda(int n, int c, int h, int w, float* src, float* pSlope, float* dst, cudaStream_t stream);
void normalizeCuda(int n, int c, int h, int w, float* src, float* pWeight, float* dst, cudaStream_t stream);
void innerScaleCuda(int n, int c, int h, int w, float* src, float* SRC, float* dst, cudaStream_t stream);
void avgChanCuda(int n, int c, int h, int w, float* src, float* dst, cudaStream_t stream);
void doCudaLoop(unsigned int* vp, int vn, cudaStream_t stream);
#endif
