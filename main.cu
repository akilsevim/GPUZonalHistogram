#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>

#include "kernel.cu"


int main(int argc, char* argv[])
{
    Timer timer;
    startTime(&timer);

    int line_count = 0;
    int histo_row_count = 64;
    int histo_col_count = 64;
    int zoom_level = 6;
    //stream number
    int stream_number = 3;
    //limit of the rows send to device
    int chunk_size = 25000;
    int line_limit = 1000000000;
    FILE *file;
    cudaError_t cuda_ret;

    const char* filename;
    const char* output_file = "out.txt";

    int debug = 0;

    if(argc == 1) {
        filename = "cemetery.csv";
    } else if(argc == 2) {
        filename = argv[1];
    } else if(argc == 3) {
        if(strcmp(argv[2], "d") == 0) {
            filename = argv[1];
            debug = 1;
        } else {
            filename = argv[1];
            zoom_level = atoi(argv[2]);
            histo_row_count = (1 << zoom_level);
            histo_col_count = (1 << zoom_level);
        }
    } else if(argc == 4) {
        filename = argv[1];
        histo_row_count = atoi(argv[2]);
        histo_col_count = atoi(argv[3]);
    } else if(argc == 6) {
        filename = argv[1];
        histo_row_count = atoi(argv[2]);
        histo_col_count = atoi(argv[3]);
        stream_number = atoi(argv[4]);
        chunk_size = atoi(argv[5]);
    }

    int histo_size = histo_row_count * histo_col_count;
    
    cudaStream_t stream[stream_number];
    for(int i = 0; i < stream_number; i++) {
        cuda_ret = cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
        if(cuda_ret != cudaSuccess) FATAL("Stream %d couldn't be created:%d",i, cuda_ret);
    }

    point* points[stream_number];
    point* points_d[stream_number];
    unsigned int *histogram[stream_number];
    unsigned int *histogram_d[stream_number];
    for(int i = 0; i < stream_number; i++) {
        //Unified Memory Implementation
        /*cuda_ret = cudaMallocManaged((void **) &points[i], sizeof(point)*chunk_size);
        if(cuda_ret != cudaSuccess) FATAL("Chunck %d couldn't be allocated:%d",i, cuda_ret);
        cuda_ret = cudaStreamAttachMemAsync(stream[i], points[i]);
        if(cuda_ret != cudaSuccess) FATAL("Chunck %d couldn't be attached:%d",i, cuda_ret);*/

        cuda_ret = cudaHostAlloc((void **) &points[i], sizeof(point)*chunk_size, cudaHostAllocDefault);
        if(cuda_ret != cudaSuccess) FATAL("Chunck %d couldn't be allocated:%d",i, cuda_ret);

        cuda_ret = cudaMalloc((void **) &points_d[i], sizeof(point)*chunk_size);
        if(cuda_ret != cudaSuccess) FATAL("Chunck %d couldn't be allocated at device:%d",i, cuda_ret);

        cuda_ret = cudaHostAlloc((void **) &histogram[i], sizeof(unsigned int)*histo_size, cudaHostAllocDefault);
        if(cuda_ret != cudaSuccess) FATAL("Histogram %d couldn't be allocated:%d",i, cuda_ret);

        cuda_ret = cudaMalloc((void **) &histogram_d[i], sizeof(unsigned int)*histo_size);
        if(cuda_ret != cudaSuccess) FATAL("Histogram %d couldn't be allocated at device:%d",i, cuda_ret);

        cuda_ret = cudaMemset(histogram_d[i], 0, histo_size * sizeof(unsigned int));
        if(cuda_ret != cudaSuccess) FATAL("Unable to set device histogram %d", i);

        cudaDeviceSynchronize();
    }

    /**********Unified Memory Related Codes**********/
    /*cuda_ret = cudaMallocManaged((void **) &histogram, sizeof(unsigned int)*histo_size);
    if(cuda_ret != cudaSuccess) FATAL("Histogram couldn't be allocated on device: %d", cuda_ret);*/
    //memset(histogram, 0, histo_size * sizeof(unsigned int));
    //float *lats_d;
    //float *lons_d;
    //unsigned int *histo_d;

    int BLOCK_SIZE = 512;
    int MAX_GRID_SIZE = 12;
    int GRID_SIZE = ((chunk_size - 1) / BLOCK_SIZE) + 1;
    GRID_SIZE = GRID_SIZE > MAX_GRID_SIZE ? MAX_GRID_SIZE : GRID_SIZE;

    dim3 DimGrid(GRID_SIZE, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    printf("\nLoading file... %s", filename);
    
    file = fopen(filename, "r");
    if(file == NULL) {
        perror("fopen()");
    }
    
    //Read and print first 20 lines for deubgging purposes
    int test_1 = 0;
    if(debug == 1) {
        char * line = NULL;
        size_t len = 0;
        ssize_t read;
        int count = 0;
        while ((read = getline(&line, &len, file)) != -1) {
            printf("%s", line);
            if(count == 20) break;
            count++;
        }
        fclose(file);
        return 0;
    }
    
    //Worl coordinate max and min values to calculate histogram
    float lat_max = 90.0f, lat_min = -90.0f, lon_max = 180.0f, lon_min = -180.0f;
    //lat_max:78.213684, lon_max:178.487671, lat_min:-54.901825, lon_min:-176.255707

    char* line;
    int end = 0;
    int chunk_counter = 0;
    int test = 0;
    unsigned int *histogram_final;
    histogram_final = (unsigned int*)malloc(histo_size * sizeof(unsigned int));
    memset(histogram_final, 0.0f, histo_size * sizeof(unsigned int));
    
    while (1) {
        for(int i = 0; i < stream_number; i++) {
            float lat, lon = lat = 0.0f;
            
            
            if(strcmp(filename, "cemetery.csv") == 0)) {
                //pattern for cemetery dataset
                test = fscanf(file, "%f,%f\n", &lon, &lat);
            } else {
                //pattern for all_nodes dataset
                test = fscanf(file, "%*s\t%f\t%f\t%*s\n", &lat, &lon);
            }

            if(test == EOF) {//If we reached EOF
                end = 1;
                break;
            } else if(test != 2) { //If there are not 2 vals matched
                i--;
                continue;
            }
            //if there is an outlier set it to the limits, we could have skip those rows also but prefferd to add them to camputation
            if(lat > lat_max) lat = lat_max;
            if(lon > lon_max) lon = lon_max;
            if(lat < lat_min) lat = lat_min;
            if(lon < lon_min) lon = lon_min;
            points[i][chunk_counter].lat = lat;
            points[i][chunk_counter].lon = lon;
            line_count++;
            
            //if a limit set check it
            if(line_limit != 0 && line_limit == line_count) {
                end = 1;
                break;
            }
        }

        chunk_counter++;

        if(chunk_counter == chunk_size || (end == 1 && chunk_counter != 1) || line_limit == line_count) {
            if(lat_max > FLT_MAX) lat_max = FLT_MAX;
            if(lon_max > FLT_MAX) lon_max = FLT_MAX;
            if(lat_min > FLT_MIN) lat_min = FLT_MIN;
            if(lon_min > FLT_MIN) lon_min = FLT_MIN;
            int sent_size;
            for(int i = 0; i < stream_number; i++) {
                sent_size = chunk_size;
                if(line_count%chunk_size != 0) sent_size = line_count%chunk_size;
                cudaMemcpyAsync(points_d[i], points[i], sent_size*sizeof(point*), cudaMemcpyHostToDevice, stream[i]);
            }
            for(int i = 0; i < stream_number; i++) {
                histogram_kernel<<<DimGrid, DimBlock, histo_size * sizeof(unsigned int), stream[i]>>>(
                    points_d[i],
                    histogram_d[i],
                    sent_size, 
                    histo_row_count, 
                    histo_col_count, 
                    lat_max, 
                    lat_min, 
                    lon_max, 
                    lon_min);
            }

            for(int i = 0; i < stream_number; i++) {
                cudaMemcpyAsync(histogram[i], histogram_d[i], histo_size*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream[i]);
            }

            for(int i = 0; i < stream_number; i++) {
                cudaStreamSynchronize(stream[i]);
            }

            for(int i = 0; i < stream_number; i++) {
                for(int j = 0; j < histo_size; j++) {
                    histogram_final[j] += histogram[i][j];
                }
            }
            
            chunk_counter = 0;

            for(int i = 0; i < stream_number; i++) {
                cuda_ret = cudaMemset(histogram_d[i], 0, histo_size * sizeof(unsigned int));
                if(cuda_ret != cudaSuccess) FATAL("Unable to set device histogram %d", i);
            }

        }
        if(end == 1) {
            break;
        }
        
    }

    fclose(file);

    FILE *out_file;
    out_file = fopen(output_file, "w");
    printf("Done!");
    for(int i = 0; i < histo_row_count; i++) {
        for(int j = 0; j < histo_col_count; j++) {
            fprintf(out_file, "%d\t%d\t%d\n", i, j, histogram_final[i * histo_col_count + j]);
            printf("\n%d\t%d\t%d", i, j, histogram_final[i * histo_col_count + j]);
        }
    }

    cudaFreeHost(points);
    cudaFreeHost(histogram);

    cudaFree(points_d);

    for(int i = 0; i < stream_number; i++) {
        cudaStreamDestroy(stream[i]);
    }
    return 0;
}
