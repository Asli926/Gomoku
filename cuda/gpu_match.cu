#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "gpu_match.cuh"
#include <stdio.h>
#include <cstdlib>

__constant__ char patterns_p1[16 * 6];
__constant__ char patterns_p2[16 * 6];
__constant__ int dfas_p1[16 * 7];
__constant__ int dfas_p2[16 * 7];
__constant__ int pattern_size[16];
__constant__ int score_map[16];

inline void checkCudaError(cudaError err, const char* loc) {
    if (err != cudaSuccess) {
        printf("[%s]CUDA runtime error: %s.\n", loc, cudaGetErrorString(err));
        fflush(stdout);
    }
}

extern "C"
int setPatternRelatedInfo(char* patterns_p1_, char* patterns_p2_, int* dfas_p1_, int* dfas_p2_,
                                  int* pattern_size_, int* score_map_) {
    checkCudaError(cudaMemcpyToSymbol(patterns_p1, &patterns_p1_, sizeof(char) * 16 * 6, 0, cudaMemcpyHostToDevice), "constant patterns1");
    checkCudaError(cudaMemcpyToSymbol(patterns_p2, &patterns_p2_, sizeof(char) * 16 * 6, 0, cudaMemcpyHostToDevice), "constant patterns2");
    checkCudaError(cudaMemcpyToSymbol(dfas_p1, &dfas_p1_, sizeof(int) * 16 * 7, 0, cudaMemcpyHostToDevice), "constant dfa1");
    checkCudaError(cudaMemcpyToSymbol(dfas_p2, &dfas_p2_, sizeof(int) * 16 * 7, 0, cudaMemcpyHostToDevice), "constant dfa2");
    checkCudaError(cudaMemcpyToSymbol(pattern_size, &pattern_size_, sizeof(int) * 16, 0, cudaMemcpyHostToDevice), "constant pattern size");
    checkCudaError(cudaMemcpyToSymbol(score_map, &score_map_, sizeof(int) * 16, 0, cudaMemcpyHostToDevice), "constant score map");

    return 0;
}

__global__
void match_count_kernel(int *res, char* lines, int* line_size, int player_num) {
    __shared__ int temp[128];
    __shared__ int result[2];
    int wtid = blockIdx.x * blockDim.x + threadIdx.x;
    temp[wtid] = 0;

    int tid = threadIdx.x;
    int threadId = tid % 16, blockId = tid / 16;

    char *pattern = patterns_p2 + 6 * threadId;
    int *nxt = dfas_p2 + 7 * threadId;

    if (blockIdx.x == 0 && player_num == 1 || blockIdx.x == 1 && player_num == 2) {
        pattern = patterns_p1 + 6 * threadId;
        nxt = dfas_p1 + 7 * threadId;
    }

    int i = 0, j;
    int m = pattern_size[threadId], n = line_size[blockId];
    char *line = lines + (20 * blockId);

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
            temp[wtid] += score_map[threadId];
            i = i - j + 1;
            continue;
        }

        // Otherwise: we have traversed to the end of 'line', so just break
        break;
    }

    __syncthreads(); // synchronize all threads

    if (wtid == 0)
    {
        int sum = 0;
        for (int t = 0; t < 64; t++)
        {
            sum += temp[t];
        }
        result[0] = sum;
    }

    if (wtid == 64) {
        int sum = 0;
        for (int t = 64; t < 128; t++)
        {
            sum += temp[t];
        }
        result[1] = sum;
    }

    *res = result[0] - result[1];
}


extern "C"
int match_count_multiple(char* lines, int* line_size, int player_num) {

    char *dev_lines;
    int *dev_line_size, *dev_res;

    /* =============== Malloc memory on GPU =============== */

    // malloc lines (4 lines of 20 characters)
    checkCudaError(cudaMalloc((void**)&dev_lines, sizeof(char) * 4 * 20), "Malloc lines");

    // malloc line_size
    checkCudaError(cudaMalloc((void**)&dev_line_size, sizeof(int) * 4), "Malloc line size");

    // malloc result
    checkCudaError(cudaMalloc((void**)&dev_res, sizeof(int)), "Malloc result");


    /* =============== Copy memory from RAM to GPU device =============== */

    checkCudaError(cudaMemcpy(dev_lines, lines, sizeof(char) * 4 * 20, cudaMemcpyHostToDevice), "copy lines");
    checkCudaError(cudaMemcpy(dev_line_size, line_size, sizeof(int) * 4, cudaMemcpyHostToDevice), "copy line size");

    int *res = (int*) malloc(sizeof(int));

    match_count_kernel<<<2, 64>>>(dev_res, dev_lines, dev_line_size, player_num);
    checkCudaError(cudaMemcpy(res, dev_res, sizeof(int), cudaMemcpyDeviceToHost), "copy result from GPU");

    int out = *res;

    /* =============== Free memory =============== */

    checkCudaError(cudaFree(dev_lines), "free lines");
    checkCudaError(cudaFree(dev_res), "free res");
    checkCudaError(cudaFree(dev_line_size), "free line size");

    free(res);

    return out;
}