#include "Cuda3DArray.h"
#include <string.h>
#include <iostream>
#include <limits>

CCuda3DArray::CCuda3DArray(int vWidth, int vHeight, int vDepth, cudaChannelFormatDesc& vElementType, bool vNormalizedCoords/*=false*/) : m_pSurfaceObject(NULL), m_pTextObject(NULL), m_pArrayElementFormat(new cudaChannelFormatDesc(vElementType))
{
	if (0 == vWidth)
		std::cout << "Invalid array width = " << vWidth << std::endl;
	m_Size = make_cudaExtent(vWidth, vHeight, vDepth);
	if (vDepth)
		exitIfCudaError(cudaMalloc3DArray(&m_pArray3D, m_pArrayElementFormat, m_Size, cudaArraySurfaceLoadStore));
	else
		exitIfCudaError(cudaMallocArray(&m_pArray3D, m_pArrayElementFormat, vWidth, vHeight));

	memset(&m_DataOfTextSurfaceLA, 0, sizeof(m_DataOfTextSurfaceLA));
	memset(&m_TextFormatFARN, 0, sizeof(m_TextFormatFARN));
	m_DataOfTextSurfaceLA.resType = cudaResourceTypeArray;
	m_TextFormatFARN.normalizedCoords = vNormalizedCoords;
	m_DataOfTextSurfaceLA.res.array.array = m_pArray3D;
}

CCuda3DArray::~CCuda3DArray()
{
	if (NULL != m_pSurfaceObject) { cudaDestroySurfaceObject(m_pSurfaceObject); m_pSurfaceObject = NULL; }
	if (NULL != m_pTextObject) { cudaDestroyTextureObject(m_pTextObject); m_pTextObject = NULL; }
	if (NULL != m_pArrayElementFormat){ delete m_pArrayElementFormat; m_pArrayElementFormat = NULL; }
	cudaFreeArray(m_pArray3D);
}

cudaSurfaceObject_t CCuda3DArray::offerSurfaceObject()
{
	if (NULL == m_pSurfaceObject) cudaCreateSurfaceObject(&m_pSurfaceObject, &m_DataOfTextSurfaceLA);
	return m_pSurfaceObject;
}

cudaTextureObject_t CCuda3DArray::offerTextureObject(bool vTexCoordsNormalized, cudaTextureFilterMode vFilterMode, cudaTextureReadMode vReadMode, cudaTextureAddressMode vOutOfRangeSolution12D, cudaTextureAddressMode vOutOfRangeSolution3D)
{
	if (m_TextFormatFARN.normalizedCoords != vTexCoordsNormalized || m_TextFormatFARN.filterMode != vFilterMode || m_TextFormatFARN.readMode != vReadMode || m_TextFormatFARN.addressMode[0] != vOutOfRangeSolution12D || m_TextFormatFARN.addressMode[1] != vOutOfRangeSolution12D || m_TextFormatFARN.addressMode[2] != vOutOfRangeSolution3D)
	{
		if (NULL != m_pTextObject) exitIfCudaError(cudaDestroyTextureObject(m_pTextObject));
		_setTextureParms(vTexCoordsNormalized, vFilterMode, vReadMode, vOutOfRangeSolution12D, vOutOfRangeSolution3D);
		exitIfCudaError(cudaCreateTextureObject(&m_pTextObject, &m_DataOfTextSurfaceLA, &m_TextFormatFARN, NULL));
	}
	return m_pTextObject;
}

void CCuda3DArray::_setTextureParms(bool vTexCoordsNormalized, cudaTextureFilterMode vFilterMode, cudaTextureReadMode vReadMode, cudaTextureAddressMode vOutOfRangeSolution12D, cudaTextureAddressMode vOutOfRangeSolution3D)
{
	if (m_Size.depth)
		m_TextFormatFARN.addressMode[2] = vOutOfRangeSolution3D;
	m_TextFormatFARN.addressMode[1] = vOutOfRangeSolution12D;
	m_TextFormatFARN.addressMode[0] = vOutOfRangeSolution12D;
	m_TextFormatFARN.normalizedCoords = vTexCoordsNormalized;
	m_TextFormatFARN.filterMode = vFilterMode;
	m_TextFormatFARN.readMode = vReadMode;
}

void CCuda3DArray::copyData2Host(void* vopHost, int vZ, cudaStream_t vStream)
{
	cudaMemcpy3DParms CopyParams = { 0 };
	CopyParams.dstPtr = make_cudaPitchedPtr(vopHost, m_Size.width*sizeof(float), m_Size.width, m_Size.height);
	CopyParams.kind = cudaMemcpyDeviceToHost;
	CopyParams.srcArray = m_pArray3D;
	CopyParams.extent = m_Size;
	if (-1 != vZ && vZ > -1 && vZ < m_Size.depth)
	{
		CopyParams.srcPos.z = vZ;
		CopyParams.extent.depth = 1;
	}
	exitIfCudaError(cudaMemcpy3DAsync(&CopyParams, vStream));
}

void CCuda3DArray::copyData2GPUArray(const void *vpHost, int vZ /*= -1*/, cudaStream_t vStream /*= NULL*/)
{
	int elementSize = (m_pArrayElementFormat->x >> 3) + (m_pArrayElementFormat->y >> 3) + (m_pArrayElementFormat->z >> 3) + (m_pArrayElementFormat->w >> 3);
	if (m_Size.depth)
	{
		cudaMemcpy3DParms CopyParams = { 0 };
		CopyParams.srcPtr = make_cudaPitchedPtr((void*)vpHost, m_Size.width*elementSize, m_Size.width, m_Size.height);
		CopyParams.kind = cudaMemcpyHostToDevice;
		CopyParams.dstArray = m_pArray3D;
		CopyParams.extent = m_Size;
		if (-1 != vZ && vZ > -1 && vZ < m_Size.depth)
		{
			CopyParams.dstPos.z = vZ;
			CopyParams.extent.depth = 1;
		}
		exitIfCudaError(cudaMemcpy3DAsync(&CopyParams, vStream));
	}
	else
	{
		exitIfCudaError(cudaMemcpyToArrayAsync(m_pArray3D, 0, 0, vpHost, m_Size.width*(m_Size.height>0 ? m_Size.height : 1)*elementSize, cudaMemcpyHostToDevice, vStream));
	}

}
