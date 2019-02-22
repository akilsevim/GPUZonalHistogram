#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "support.h"
#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;
    startTime(&timer);

    int c;
    int line_count = 0;
    int bin_size = 256;
    int zoom_level = 4;
    FILE *file;
    cudaError_t cuda_ret;

    const char* filename;

    if(argc == 1) {
        filename = "cemetery.csv";
    } else if(argc == 2) {
        filename = argv[1];
    } else if(argc == 3) {
        filename = argv[1];
        zoom_level = atoi(argv[2]);
        bin_size = (1 << zoom_level) * (1 << zoom_level);
    }

    printf("\nLoading file...");

    file = fopen(filename, "r");
    if (file) {
        while ((c = getc(file)) != EOF)
            if(c == '\n') line_count++;
    }

    float *lats_h = (float*) malloc(sizeof(float)*line_count);
    float *lons_h = (float*) malloc(sizeof(float)*line_count);

    unsigned int *bin_h = (unsigned int*) malloc(sizeof(unsigned int)*bin_size); 

    float *lats_d;
    float *lons_d;
    unsigned int *bin_d;

    int i = 0;
    rewind(file);
    while (i < line_count) {
        fscanf(file, "%f,", &lats_h[i]);
        fscanf(file, "%f", &lons_h[i]);
        i++;
    }

    fclose(file);

    printf("\n%d points loaded. ", line_count);
    stopTime(&timer); printf("%f s", elapsedTime(timer));

    printf("\nAllocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((float**)&lats_d, line_count * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((float**)&lons_d, line_count * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((unsigned int**)&bin_d, bin_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    
    printf("\nCopying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(lats_d, lats_h, line_count * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemcpy(lons_d, lons_h, line_count * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemset(bin_d, 0, bin_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    histogram(lats_d, lons_d, bin_d, line_count, bin_size);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(bin_h, bin_d, bin_size * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    cudaFree(lats_d); cudaFree(lons_d);
    free(lats_h); free(lons_h);

    return 0;
}
