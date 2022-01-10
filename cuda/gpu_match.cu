#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "gpu_match.cuh"
#include <stdio.h>

inline void checkCudaError(cudaError err) {
    if (err != cudaSuccess) {
        printf("CUDA runtime error: %s.\n", cudaGetErrorString(err));
        fflush(stdout);
    }
}

__global__
void match_count_kernel(int *res, char* lines, char* patterns, int* dfas, int* pattern_size, int* line_size, int* score_map) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int i = 0, j;
    int m = pattern_size[threadIdx.x], n = line_size[blockIdx.x];
    char *line = lines + (20 * blockIdx.x);
    char *pattern = patterns + 6 * threadIdx.x;
    int *nxt = dfas + 7 * threadIdx.x;

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
            res[tid] += score_map[threadIdx.x];
            i = i - j + 1;
            continue;
        }

        // Otherwise: we have traversed to the end of 'line', so just break
        break;
    }
}


extern "C"
int match_count_multiple(char* lines, char* patterns, int* dfas, int* pattern_size, int* line_size, int* score_map) {

    char *dev_lines, *dev_patterns;
    int *dev_dfas;
    int *dev_pattern_size, *dev_line_size, *dev_score_map, *dev_res;

    /* =============== Malloc memory on GPU =============== */

    // malloc lines (4 lines of 20 characters)
    checkCudaError(cudaMalloc((void**)&dev_lines, sizeof(char) * 4 * 20));

    // malloc patterns (16 patterns of 6 characters)
    checkCudaError(cudaMalloc((void**)&dev_patterns, sizeof(char) * 16 * 6));

    // malloc dfas (16 dfa arrays of 7 integers)
    checkCudaError(cudaMalloc((void**)&dev_dfas, sizeof(int) * 16 * 7));

    // malloc pattern_size
    checkCudaError(cudaMalloc((void**)&dev_pattern_size, sizeof(int) * 16));

    // malloc line_size
    checkCudaError(cudaMalloc((void**)&dev_line_size, sizeof(int) * 4));

    // malloc score_map
    checkCudaError(cudaMalloc((void**)&dev_score_map, sizeof(int) * 16));

    // malloc result
    checkCudaError(cudaMalloc((void**)&dev_res, sizeof(int) * 64));


    /* =============== Copy memory from RAM to GPU device =============== */

    checkCudaError(cudaMemcpy(dev_lines, lines, sizeof(char) * 4 * 20, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_patterns, patterns, sizeof(char) * 4 * 6, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_dfas, dfas, sizeof(int) * 16 * 7, cudaMemcpyHostToDevice));

    checkCudaError(cudaMemcpy(dev_pattern_size, pattern_size, sizeof(int) * 16, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_line_size, line_size, sizeof(int) * 4, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dev_score_map, score_map, sizeof(int) * 16, cudaMemcpyHostToDevice));

    int res[64];
    for (int k = 0; k < 64; k ++) res[k] = 0;
    checkCudaError(cudaMemcpy(dev_res, res, sizeof(int) * 64, cudaMemcpyHostToDevice));

    int out = 0;
    match_count_kernel<<<4, 16>>>(dev_res, dev_lines, dev_patterns, dev_dfas, dev_pattern_size, dev_line_size, dev_score_map);
    checkCudaError(cudaMemcpy(res, dev_res, sizeof(int) * 64, cudaMemcpyDeviceToHost));

    // reduce process
    for (int k = 0; k < 64; k ++) out += res[k];
    if (out != 0) {
        printf("out: %d\n", out);
        fflush(stdout);
    }

    /* =============== Copy memory from RAM to GPU device =============== */

    checkCudaError(cudaFree(dev_lines));
    checkCudaError(cudaFree(dev_patterns));
    checkCudaError(cudaFree(dev_dfas));
    checkCudaError(cudaFree(dev_res));
    checkCudaError(cudaFree(dev_line_size));
    checkCudaError(cudaFree(dev_pattern_size));
    checkCudaError(cudaFree(dev_score_map));

    return out;
}