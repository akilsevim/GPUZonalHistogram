#define BLOCK_SIZE 512 

__global__ void histogram_kernel(
    float* lats, 
    float* lons, 
    unsigned int* histo_d,
    unsigned int line_count,
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
    while(i < line_count) 
    {
        int col = (lats[i] - lat_min) * histo_col_count / (lat_max - lat_min);
        int row = (lons[i] - lon_min) * histo_row_count / (lon_max - lon_min);
        atomicAdd(&(histogram_private[row * histo_col_count + col]), 1);
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();
    //Compute the global histogram
    i = 0;
    while((blockDim.x * i + threadIdx.x) < histo_size) 
    {
        atomicAdd(&(histo_d[blockDim.x * i + threadIdx.x]), histogram_private[blockDim.x * i + threadIdx.x]);
        i++;
    }
}

void histogram(float* lats, 
    float* lons, 
    unsigned int* histo_d, 
    unsigned int line_count,
    unsigned int histo_row_count, 
    unsigned int histo_col_count,
    float lat_max, 
    float lat_min, 
    float lon_max, 
    float lon_min) {

    int MAX_GRID_SIZE = 12;
    int GRID_SIZE = ((line_count - 1) / BLOCK_SIZE) + 1;
    GRID_SIZE = GRID_SIZE > MAX_GRID_SIZE ? MAX_GRID_SIZE : GRID_SIZE;

    dim3 DimGrid(GRID_SIZE, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    histogram_kernel<<<DimGrid, DimBlock, histo_col_count * histo_row_count * sizeof(unsigned int)>>>(
        lats,
        lons, 
        histo_d,
        line_count, 
        histo_row_count, 
        histo_col_count, 
        lat_max, 
        lat_min, 
        lon_max, 
        lon_min);

}