#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

__global__ void match_count_kernel(int *res, char** lines, char** patterns, int** dfas, int* pattern_size, int* line_size, int* score_map) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = 0, j;
    int m = pattern_size[threadIdx.x], n = line_size[blockIdx.x];
    char *line = lines[blockIdx.x];
    char *pattern = patterns[threadIdx.x];
    int *nxt = dfas[threadIdx.x];

    // i is the pointer of 'line'
    // j is the pointer of 'pattern'
    while (i < n) {

        // start a single search (first set j to 0)
        j = 0;
        while (i < n && j < m) {
            if (j == -1 || line[i] == pattern[j]) {
                i++;
                j++;
            } else {
                j = nxt[j];
            }
        }

        // right after a single search
        // If j == m: we have found one match, so res ++
        if (j == m) {
            res[tid] ++;
            i = i - j + 1;
            continue;
        }

        // Otherwise: we have traversed to the end of 'line', so just break
        break;
    }
}


extern "C" int match_count_multiple(char** lines, char** patterns, int** dfas, int* pattern_size, int* line_size, int* score_map) {

    char **dev_lines, **dev_patterns;
    int** dev_dfas;
    int *dev_pattern_size, *dev_line_size, *dev_score_map, *dev_res;

    /* =============== Malloc memory on GPU =============== */

    // malloc lines
    cudaMalloc((void**)&dev_lines, sizeof(char*) * 4);
    for (int k = 0; k < 4; k ++) {
        cudaMalloc((void **) &(dev_lines[k]), sizeof(char) * line_size[k]);
    }

    // malloc patterns
    cudaMalloc((void**)&dev_patterns, sizeof(char*) * 16);
    for (int k = 0; k < 16; k ++) {
        cudaMalloc((void **) &(dev_patterns[k]), sizeof(char) * pattern_size[k]);
    }

    // malloc dfas
    cudaMalloc((void**)&dev_dfas, sizeof(int*) * 16);
    for (int k = 0; k < 16; k ++) {
        cudaMalloc((void **) &(dev_dfas[k]), sizeof(int) * pattern_size[k]);
    }

    // malloc pattern_size
    cudaMalloc((void**)&dev_pattern_size, sizeof(int) * 16);

    // malloc line_size
    cudaMalloc((void**)&dev_line_size, sizeof(int) * 4);

    // malloc score_map
    cudaMalloc((void**)&dev_score_map, sizeof(int) * 16);

    // malloc result
    cudaMalloc((void**)&dev_res, sizeof(int) * 64);


    /* =============== Copy memory from RAM to GPU device =============== */

    cudaMemcpy(dev_lines, lines, sizeof(char*) * 4, cudaMemcpyHostToDevice);
    for (int k = 0; k < 4; k ++) {
        cudaMemcpy(dev_lines[k], lines[k], sizeof(char) * line_size[k], cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dev_patterns, patterns, sizeof(char*) * 4, cudaMemcpyHostToDevice);
    for (int k = 0; k < 4; k ++) {
        cudaMemcpy(dev_lines[k], lines[k], sizeof(char) * pattern_size[k], cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dev_dfas, dfas, sizeof(int*) * 16, cudaMemcpyHostToDevice);
    for (int k = 0; k < 16; k ++) {
        cudaMemcpy(dev_dfas[k], dfas[k], sizeof(int) * (pattern_size[k] + 1), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(dev_pattern_size, pattern_size, sizeof(int) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_line_size, line_size, sizeof(int) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_score_map, score_map, sizeof(int) * 16, cudaMemcpyHostToDevice);

    int res[64];
    for (int k = 0; k < 64; k ++) res[k] = 0;
    cudaMemcpy(dev_res, res, sizeof(int) * 64, cudaMemcpyHostToDevice);

    int out = 0;
    match_count_kernel<<<4, 16>>>(dev_res, dev_lines, dev_patterns, dev_dfas, dev_pattern_size, dev_line_size, dev_score_map);
    cudaMemcpy(res, dev_res, sizeof(int) * 64, cudaMemcpyDeviceToDevice);

    // reduce process
    for (int k = 0; k < 64; k ++) out += res[k];

    /* =============== Copy memory from RAM to GPU device =============== */

    for (int k = 0; k < 4; k ++) {
        cudaFree(dev_lines[k]);
    }

    for (int k = 0; k < 16; k ++) {
        cudaFree(dev_patterns[k]);
        cudaFree(dev_dfas[k]);
    }

    cudaFree(dev_res);
    cudaFree(dev_line_size);
    cudaFree(dev_pattern_size);
    cudaFree(dev_score_map);

    return out;
}