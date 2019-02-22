/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to
#define BLOCK_SIZE 512 

// INSERT KERNEL(S) HERE
__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) 
{
    //Dynamic Shared Memory
    extern __shared__ unsigned int histo_private[];
    
    //Initialize private hisogram
    int i = 0;
    while((blockDim.x * i + threadIdx.x) < num_bins) 
    {
        histo_private[blockDim.x * i + threadIdx.x] = 0.0f;
        i++;
    }
    __syncthreads();
    //Compute the private histograms
    i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < num_elements) 
    {
        atomicAdd(&(histo_private[input[i]]), 1);
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();
    //Compute the global histogram
    i = 0;
    while((blockDim.x * i + threadIdx.x) < num_bins) 
    {
        atomicAdd(&(bins[blockDim.x * i + threadIdx.x]), histo_private[blockDim.x * i + threadIdx.x]);
        i++;
    }
}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    // INSERT CODE HERE
    //I tested the program on docker and find out this is the maxiumum optimal number for the grid size (N = 1m) since we have max 1536 threads in an SM
    int MAX_GRID_SIZE = 12;
    int GRID_SIZE = ((num_elements - 1) / BLOCK_SIZE) + 1;
    GRID_SIZE = GRID_SIZE > MAX_GRID_SIZE ? MAX_GRID_SIZE : GRID_SIZE;

    dim3 DimGrid(GRID_SIZE, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    histo_kernel<<<DimGrid, DimBlock, num_bins * sizeof(unsigned int)>>>(input, bins, num_elements, num_bins);

}


