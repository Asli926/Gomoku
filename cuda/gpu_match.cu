#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "gpu_match.cuh"
#include <stdio.h>
#include <cstdlib>

inline void checkCudaError(cudaError err, const char* loc) {
    if (err != cudaSuccess) {
        printf("[%s]CUDA runtime error: %s.\n", loc, cudaGetErrorString(err));
        fflush(stdout);
    }
}

__global__
void match_count_kernel(int *res, char* lines, char* patterns, int* dfas, int* pattern_size, int* line_size, int* score_map) {
    __shared__ int temp[64];
    int tid = threadIdx.x;
    temp[tid] = 0;

    int i = 0, j;
    int threadId = tid % 16, blockId = tid / 16;
    int m = pattern_size[threadId], n = line_size[blockId];
    char *line = lines + (20 * blockId);
    char *pattern = patterns + 6 * threadId;
    int *nxt = dfas + 7 * threadId;

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
            temp[tid] += score_map[threadId];
            i = i - j + 1;
            continue;
        }

        // Otherwise: we have traversed to the end of 'line', so just break
        break;
    }

    __syncthreads(); // synchronize all threads

    if (tid == 0)
    {
        int sum = 0;
        for (int t = 0; t < 64; t++)
        {
            sum += temp[t];
        }
        *res = sum;
    }

}


extern "C"
int match_count_multiple(char* lines, char* patterns, int* dfas, int* pattern_size, int* line_size, int* score_map) {

    char *dev_lines, *dev_patterns;
    int *dev_dfas;
    int *dev_pattern_size, *dev_line_size, *dev_score_map, *dev_res;

    /* =============== Malloc memory on GPU =============== */

    // malloc lines (4 lines of 20 characters)
    checkCudaError(cudaMalloc((void**)&dev_lines, sizeof(char) * 4 * 20), "Malloc lines");

    // malloc patterns (16 patterns of 6 characters)
    checkCudaError(cudaMalloc((void**)&dev_patterns, sizeof(char) * 16 * 6), "Malloc patterns");

    // malloc dfas (16 dfa arrays of 7 integers)
    checkCudaError(cudaMalloc((void**)&dev_dfas, sizeof(int) * 16 * 7), "Malloc dfas");

    // malloc pattern_size
    checkCudaError(cudaMalloc((void**)&dev_pattern_size, sizeof(int) * 16), "Malloc pattern size");

    // malloc line_size
    checkCudaError(cudaMalloc((void**)&dev_line_size, sizeof(int) * 4), "Malloc line size");

    // malloc score_map
    checkCudaError(cudaMalloc((void**)&dev_score_map, sizeof(int) * 16), "Malloc score map");

    // malloc result
    checkCudaError(cudaMalloc((void**)&dev_res, sizeof(int)), "Malloc result");


    /* =============== Copy memory from RAM to GPU device =============== */

    checkCudaError(cudaMemcpy(dev_lines, lines, sizeof(char) * 4 * 20, cudaMemcpyHostToDevice), "copy lines");
    checkCudaError(cudaMemcpy(dev_patterns, patterns, sizeof(char) * 6 * 16, cudaMemcpyHostToDevice), "copy patterns");
    checkCudaError(cudaMemcpy(dev_dfas, dfas, sizeof(int) * 16 * 7, cudaMemcpyHostToDevice), "copy dfas");

    checkCudaError(cudaMemcpy(dev_pattern_size, pattern_size, sizeof(int) * 16, cudaMemcpyHostToDevice), "copy pattern size");
    checkCudaError(cudaMemcpy(dev_line_size, line_size, sizeof(int) * 4, cudaMemcpyHostToDevice), "copy line size");
    checkCudaError(cudaMemcpy(dev_score_map, score_map, sizeof(int) * 16, cudaMemcpyHostToDevice), "copy score map");

    int *res = (int*) malloc(sizeof(int));

    match_count_kernel<<<1, 64>>>(dev_res, dev_lines, dev_patterns, dev_dfas, dev_pattern_size, dev_line_size, dev_score_map);
//    cudaDeviceSynchronize();
    checkCudaError(cudaMemcpy(res, dev_res, sizeof(int), cudaMemcpyDeviceToHost), "copy result from GPU");
    int out = *res;
//    if (out != 0) {
//        printf("out: %d\n", out);
//        fflush(stdout);
//    }

    /* =============== Free memory =============== */

    checkCudaError(cudaFree(dev_lines), "free lines");
    checkCudaError(cudaFree(dev_patterns), "free patterns");
    checkCudaError(cudaFree(dev_dfas), "free dfas");
    checkCudaError(cudaFree(dev_res), "free res");
    checkCudaError(cudaFree(dev_line_size), "free line size");
    checkCudaError(cudaFree(dev_pattern_size), "free pattern size");
    checkCudaError(cudaFree(dev_score_map), "free score map");

    free(res);

    return out;
}