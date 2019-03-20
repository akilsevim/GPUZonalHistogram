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
    int histo_row_count = 256;
    int histo_col_count = 256;
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
        histo_row_count = (1 << zoom_level);
        histo_col_count = (1 << zoom_level);
    } else if(argc == 4) {
        filename = argv[1];
        histo_row_count = atoi(argv[2]);
        histo_col_count = atoi(argv[3]);
    }

    int histo_size = histo_row_count * histo_col_count;

    printf("\nLoading file...");

    file = fopen(filename, "r");
    if (file) {
        while ((c = getc(file)) != EOF)
            if(c == '\n') line_count++;
    }

    float *lats_h = (float*) malloc(sizeof(float)*line_count);
    float *lons_h = (float*) malloc(sizeof(float)*line_count);

    unsigned int * histo_h = (unsigned int*) malloc(sizeof(unsigned int) * histo_size); 

    float *lats_d;
    float *lons_d;
    unsigned int *histo_d;

    int i = 0;
    rewind(file);
    float lat_max = -256.0f, lat_min = 256.0f, lon_max = -256.0f, lon_min = 256.0f;

    while (i < line_count) {
        fscanf(file, "%f,", &lats_h[i]);
        fscanf(file, "%f", &lons_h[i]);
        if(lats_h[i] > lat_max) lat_max = lats_h[i];
        if(lons_h[i] > lon_max) lon_max = lons_h[i];
        if(lats_h[i] < lat_min) lat_min = lats_h[i];
        if(lons_h[i] < lon_min) lon_min = lons_h[i];
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
    cuda_ret = cudaMalloc((unsigned int**)&histo_d, histo_size * sizeof(unsigned int));
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

    cuda_ret = cudaMemset(histo_d, 0, histo_size * sizeof(unsigned int));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    histogram(lats_d, lons_d, histo_d, line_count, histo_row_count, histo_col_count, lat_max, lat_min, lon_max, lon_min);
    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(histo_h, histo_d, histo_size * sizeof(unsigned int),
        cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    for(int i = 0; i < histo_row_count; i++) {
        for(int j = 0; j < histo_col_count; j++) {
            printf("%d %d: %d\t", i, j, histo_h[i * histo_col_count + j]);
        }
        printf("\n");
    }

    cudaFree(lats_d); cudaFree(lons_d);
    free(lats_h); free(lons_h);

    return 0;
}