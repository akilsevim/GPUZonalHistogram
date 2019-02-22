#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
    int c;
    int lineCount = 0;
    FILE *file;
    file = fopen("cemetery.csv", "r");
    if (file) {
        while ((c = getc(file)) != EOF)
            if(c == '\n') lineCount++;
    }
    printf("%d points loaded \n", lineCount);

    float *lats = (float*) malloc(sizeof(float)*lineCount);
    float *lons = (float*) malloc(sizeof(float)*lineCount);

    int i = 0;
    rewind(file);
    while (i < lineCount) {
        fscanf(file, "%f,", &lats[i]);
        fscanf(file, "%f", &lons[i]);
        i++;
    }

    i = 0;
    while (i < lineCount) {
        printf("lat:%f, lon:%f",lats[i], lons[i]);
        i++;
    }


        
    fclose(file);



    return 0;
}
