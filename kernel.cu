#include "support.h"

__global__ void histogram_kernel(
    point* points,
    unsigned int* histogram,
    unsigned int chunk_size,
    unsigned int histo_row_count, 
    unsigned int histo_col_count,
    float lat_max, 
    float lat_min, 
    float lon_max, 
    float lon_min) 
{
    //Dynamic Shared Memory
    extern __shared__ unsigned int histogram_private[];

    unsigned int histo_size = histo_col_count * histo_row_count;
    
    //Initialize private hisogram
    int i = 0;
    while((blockDim.x * i + threadIdx.x) < histo_size) 
    {
        histogram_private[blockDim.x * i + threadIdx.x] = 0.0f;
        i++;
    }
    __syncthreads();

    //Compute the private histograms
    i = blockDim.x * blockIdx.x + threadIdx.x;
    while(i < chunk_size) 
    {
        int col = (points[i].lat - lat_min) * histo_col_count / (lat_max - lat_min);
        int row = (points[i].lon - lon_min) * histo_row_count / (lon_max - lon_min);
        if(col > histo_col_count) col = histo_col_count - 1;
        if(row > histo_row_count) row = histo_row_count - 1;
        atomicAdd(&(histogram_private[row * histo_col_count + col]), 1);
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();
    //Compute the global histogram
    i = 0;
    while((blockDim.x * i + threadIdx.x) < histo_size) 
    {
        atomicAdd(&(histogram[blockDim.x * i + threadIdx.x]), histogram_private[blockDim.x * i + threadIdx.x]);
        i++;
    }
}


