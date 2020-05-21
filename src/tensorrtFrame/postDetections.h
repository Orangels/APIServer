#pragma once
struct SFloat7
{
	float f[7];
};
void postDetections(int vBatchSize, int vHeight, int vWidth, float* viopDetections, int vkeep_top_k=100, cudaStream_t vStream=0);
