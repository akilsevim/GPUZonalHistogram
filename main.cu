#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "support.h"

int main(int argc, char* argv[])
{
    Timer timer;
    startTime(&timer);

    int c;
    int line_count = 0;
    FILE *file;
    cudaError_t cuda_ret;

    const char* filename;

    if(argc == 1) {
        filename = "cemetery.csv";
    } else if(argc == 2) {
        filename = argv[1];
    }

    printf("\nLoading file...");

    file = fopen(filename, "r");
    if (file) {
        while ((c = getc(file)) != EOF)
            if(c == '\n') line_count++;
    }

    float *lats_h = (float*) malloc(sizeof(float)*line_count);
    float *lons_h = (float*) malloc(sizeof(float)*line_count);

    float *lats_d;
    float *lons_d;

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

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((float**)&lats_d, line_count * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((float**)&lons_d, line_count * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(lats_d, lats_h, line_count * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemcpy(lons_d, lons_h, line_count * sizeof(float),
        cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    cudaFree(lats_d); cudaFree(lons_d);
    free(lats_h); free(lons_h);

    return 0;
}
