/*
 * This program uses the host CURAND API to generate 2^30 normally distributed floats
 * then bins them into 10,000 bins using both CPU and GPU methods.
 */

//Standard C Libraries
#include <stdio.h>						//Printf
#include <stdlib.h>
#include <math.h>						//floor
#include <conio.h>						//getch
#include <time.h>						//time(NULL)

//Cuda Libraries
#include <cuda.h>
#include <curand.h>						//Cuda Rand
#include <cuda_runtime.h>
#include <device_launch_parameters.h>	//ThreadIdx

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n",__FILE__,__LINE__);\
	return EXIT_FAILURE;}} while(0)

/* Cuda kernel to bin the random numbers on the GPU */
__global__ void histGPU(float* d_numbers, int* d_hist, int n, int bins)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < n)
	{
		/* calculate bin position */
		int a = floor(d_numbers[id] * bins + 0.5) + bins / 2;

		/* if the position it containded within bounds add to bin*/
		if(a > 0 && a < bins)
			atomicAdd(&d_hist[a], 1);
	}
}

int main(int argc, char *argv[])
{
	/* General variable declarations */
	int totalToGenerate = 134217728;
	curandGenerator_t gen;
	const int bins = 10000;

	/* Pointer / array declarations  [device (d_*) host (h_*)]*/
	float *d_RandomNumbers, *h_RandomNumbers;
	int *d_hist;

	// Histogram storage on host, initialised to zero (0)
	int h_hist[bins] = {0};
	int h_histCPU[bins] = {0};

	/* Timer Declarations */

	// Device
	cudaEvent_t startGPU, stopGPU;
	cudaEventCreate(&startGPU);
	cudaEventCreate(&stopGPU);
	float elapsedTimeGPU;

	// Host
	clock_t startCPU, stopCPU;
	float elapsedTimeCPU;

	printf("Allocating memory on host and device\n");
	/* Allocate n floats on host */
	h_RandomNumbers = (float *)calloc(totalToGenerate, sizeof(float));

	/* Allocate n floats on device */
	CUDA_CALL(cudaMalloc((void **)&d_RandomNumbers, totalToGenerate*sizeof(float)));
	CUDA_CALL(cudaMalloc((void **)&d_hist, totalToGenerate*sizeof(int)));

	/* Initialise histogram on the device to zero (0) */
	CUDA_CALL(cudaMemset(d_hist, 0, bins*sizeof(int)));
	
	/* Create pseudo-random number generator */
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A));
	
	/* Set seed for the random number generator based on current time */
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

	printf("Generating %i random numbers\n", totalToGenerate);
	/* Generate n floats on device */	
	CURAND_CALL(curandGenerateNormal(gen, d_RandomNumbers, totalToGenerate, 0.0f, 0.1f));

	printf("Binning generated numbers on the GPU\n");
	/* Bin the random numbers on the GPU */
	cudaEventRecord(startGPU, 0);

	histGPU<<< totalToGenerate/1024 + 1, 1024>>>(d_RandomNumbers, d_hist, totalToGenerate, bins);

	cudaEventRecord(stopGPU, 0);
	cudaEventSynchronize(stopGPU);

	cudaEventElapsedTime(&elapsedTimeGPU, startGPU, stopGPU);

	printf("Transfering numbers to the host device\n");
	/* Copy device memory to host */
	CUDA_CALL(cudaMemcpy(h_hist, d_hist, bins * sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(h_RandomNumbers, d_RandomNumbers, totalToGenerate * sizeof(float), cudaMemcpyDeviceToHost));

	printf("Binning generated numbers on the CPU\n");
	/* Bin the random numbers on the CPU */
	startCPU = clock();
	for(int i = 0; i < totalToGenerate; i++)
	{
		int e = floor(h_RandomNumbers[i] * bins + 0.5f) + bins/2;
		if( e > 0 && e < bins)
			h_histCPU[e]++;
	}
	stopCPU = clock();
	elapsedTimeCPU = (stopCPU - startCPU)/ CLOCKS_PER_SEC;

	/* Print the times taken to bin the data */
	
	printf("\nTime taken to bin the data on the GPU: %fms\n", elapsedTimeGPU);
	printf("Time taken to bin the data on the CPU: %fs\n", elapsedTimeCPU);

	/* Cleanup */
	CURAND_CALL(curandDestroyGenerator(gen));
	CUDA_CALL(cudaFree(d_RandomNumbers));
	CUDA_CALL(cudaFree(d_hist));
	free(h_RandomNumbers);    

	getch();
	return EXIT_SUCCESS;
}