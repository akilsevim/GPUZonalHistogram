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
    int histo_row_count = 16;
    int histo_col_count = 16;
    int zoom_level = 4;
    //stream number
    int stream_number = 4;
    //limit of the rows send to device
    int chunk_size = 10000;
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


    /*fseek(file, 0L, SEEK_END); 
  
    // calculating the size of the file 
    long int res = ftell(file);
    printf("\nFile Size:%d", res);


    rewind(file);
    if (file) {
        while ((c = getc(file)) != EOF)
            if(c == '\n') line_count++;
    }*/

    
    point* points[stream_number];
    point* points_d[stream_number];
    unsigned int *histogram[stream_number];
    unsigned int *histogram_d[stream_number];
    for(int i = 0; i < stream_number; i++) {
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

        cuda_ret = cudaMemset(histogram_d[i], 0.0f, histo_size * sizeof(unsigned int));
        if(cuda_ret != cudaSuccess) FATAL("Unable to set device histogram %d", i);

        cudaDeviceSynchronize();
    }


    
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

    dim3 DimGrid(1, 1, 1);
    dim3 DimBlock(1, 1, 1);
    printf("\nLoading file... %s", filename);
    
    file = fopen(filename, "r");
    if(file == NULL) {
        perror("fopen()");
    }
    int test_1 = 0;
    
    if(debug == 1) {
        float lat, lon;
        for(int i = 0; i < 10; i++) {
            
            //test_1 = fscanf(file, "%*s\t%f\t%f\t%*s\n", &lat, &lon);
            test_1 = fscanf(file, "%f,%f\n", &lat, &lon);
            
            printf("\nlat:%f, lon:%f", lat, lon);
        }
        fclose(file);
        return 0;
    }
    
    float lat_max = -256.0f, lat_min = 256.0f, lon_max = -256.0f, lon_min = 256.0f;

    
    char* line;
    int end = 0;
    int chunk_counter = 0;
    int test = 0;

    unsigned int *histogram_final;
    histogram_final = (unsigned int*)malloc(histo_size * sizeof(unsigned int));
    memset(histogram_final, 0.0f, histo_size * sizeof(unsigned int));
    
    while (1) {
        //printf("\n%d",chunk_counter);
        
        for(int i = 0; i < stream_number; i++) {
            float lat, lon = lat = 0.0f;
            
            test = fscanf(file, "%f,%f\n", &lat, &lon);
            //test = fscanf(file, "%*s\t%f\t%f\t%*s\n", &lat, &lon);

            //printf("\nlat:%f, lon:%f", lat, lon);
            if(test == EOF) {//If we reached EOF
                end = 1;
                break;
            } else if(test != 2) { //If there are not 2 vals matched
                i--;
                continue;
            }
            if(lat > lat_max) lat_max = lat;
            if(lon > lon_max) lon_max = lon;
            if(lat < lat_min) lat_min = lat;
            if(lon < lon_min) lon_min = lon;
            points[i][chunk_counter].lat = lat;
            points[i][chunk_counter].lon = lon;
            //printf("\nS==>Lat:%f, Lon:%f",points[i][chunk_counter].lat, points[i][chunk_counter].lon);

            line_count++;
        }

        //printf("\nlat_max:%f, lon_max:%f, lat_min:%f, lon_min:%f", lat_max, lon_max, lat_min, lon_min);

        chunk_counter++;

        if(chunk_counter == chunk_size || (end == 1 && chunk_counter != 1)) {

            /*for(int i = 0; i < stream_number; i++) {
                printf("Stream 1:");
                for(int j = 0; j < chunk_size; j++) {
                    printf("\nS==>Lat:%f, Lon:%f",points[i][j].lat, points[i][j].lon);
                }
                
            }*/

            if(lat_max > FLT_MAX) lat_max = FLT_MAX;
            if(lon_max > FLT_MAX) lon_max = FLT_MAX;
            if(lat_min > FLT_MIN) lat_min = FLT_MIN;
            if(lon_min > FLT_MIN) lon_min = FLT_MIN;

            for(int i = 0; i < stream_number; i++) {
                int sent_size = chunk_size;
                if(line_count%chunk_size != 0) sent_size = line_count%chunk_size;
                printf("\nSENT SIZE:%d",sent_size);
                cudaMemcpyAsync(points_d[i], points[i], sent_size*sizeof(point), cudaMemcpyHostToDevice, stream[i]);
                    
            }
            for(int i = 0; i < stream_number; i++) {
                printf("\nKernel Call:%d, LineCounter:%d, Chunk Counter:%d",i,line_count, chunk_counter);
                histogram_kernel<<<DimGrid, DimBlock, histo_size * sizeof(unsigned int), stream[i]>>>(
                    points_d[i],
                    histogram_d[i],
                    chunk_size, 
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
                    printf("\nHisto[%d][%d]:%d",i,j, histogram[i][j]);
                    histogram_final[j] += histogram[i][j];
                }
            }

            for(int i = 0; i < stream_number; i++) {
                memset(&points[i]->lat, 0.0f, sizeof(float)*chunk_size);
                memset(&points[i]->lon, 0.0f, sizeof(float)*chunk_size);
            }
            
            chunk_counter = 0;

        }
        if(end == 1) {
            break;
        }
        
    }

    /*for(int k = 0; k < histo_col_count; k++) {
        for(int l = 0; l < histo_row_count; l++) {
            
            printf("\n%d\t%d\t%d", k, l, histogram[0][k * histo_col_count + l]);
        }
    }*/
    
    //printf("lat_max:%f, lon_max:%f, lat_min:%f, lon_min:%f\n", lat_max, lon_max, lat_min, lon_min);

    
    
    


    fclose(file);

    FILE *out_file;
    out_file = fopen(output_file, "w");

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
