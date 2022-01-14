#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "gpu_match.cuh"
#include <stdio.h>
#include <cstdlib>

#define BOARD_SIZE 15
#define INT2CHAR(X) ((char)((X) + (int)'0'))


inline void checkCudaError(cudaError err, const char* loc) {
    if (err != cudaSuccess) {
        printf("[%s]CUDA runtime error: %s.\n", loc, cudaGetErrorString(err));
        fflush(stdout);
    }
}


__global__
void heuristic_moves_kernel(int *res, int *next_locations, int *moves_count,
                     int* board, int player_num,
                     char* patterns_p1, char* patterns_p2,
                     int* dfas_p1, int* dfas_p2,
                     int* pattern_size, int* score_map,
                     int max_layer) {
    int tid = threadIdx.x;

    __shared__ char next_move_masks[BOARD_SIZE * BOARD_SIZE];
    __shared__ int num_next_loc;
    __shared__ char lines[60 * 80];
    __shared__ int line_size[60 * 4];
    __shared__ int tid_score[128];
    __shared__ int score[60];
    __shared__ int const_board[BOARD_SIZE * BOARD_SIZE];

    // ================== get all available moves ==================
    if (tid < BOARD_SIZE * BOARD_SIZE) {

        // first all threads will set the mask to 0, denoting not a feasible next move
        next_move_masks[tid] = '0';

        int r = tid / BOARD_SIZE;
        int c = tid % BOARD_SIZE;

        // only no chess at this location will be considered
        if (board[tid] == 0) {
            if ((r - 1 >= 0 && board[(r - 1) * BOARD_SIZE + c] != 0)||
                    (r + 1 < BOARD_SIZE && board[(r + 1) * BOARD_SIZE + c] != 0)||
                    (c - 1 >= 0 && board[r * BOARD_SIZE + c - 1] != 0) ||
                    (c + 1 >= 0 && board[r * BOARD_SIZE + c + 1] != 0)) {
                next_move_masks[tid] = '1';
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        num_next_loc = 0;
        for (int i = 0; i < BOARD_SIZE * BOARD_SIZE && num_next_loc < 60; i ++) {
            if (next_move_masks[i] == '1') {
                next_locations[num_next_loc++] = i;
            }
        }
    }

    __syncthreads();

    // ================== get all lines according to the moves ==================
    if (tid < num_next_loc) {
        int t_plines_ini = tid * 80;

        for (int i = t_plines_ini; i < t_plines_ini + 80; i ++) {
            lines[i] = '0';
        }

        int t_size_ini = tid * 4;
        int pLines = t_plines_ini;
        int r = next_locations[tid] / BOARD_SIZE, c = next_locations[tid] % BOARD_SIZE;

        for (int i = 0; i < BOARD_SIZE; i ++) {
            lines[pLines++] = INT2CHAR(board[i * BOARD_SIZE + c]);
        }
        line_sizes[t_size_ini] = pLines; pLines = t_plines_ini + 20;

        for (int j = 0; j < board_size; j ++) {
            lines[pLines++] = INT2CHAR(board[r * BOARD_SIZE + j]);
        }
        line_sizes[t_size_ini + 1] = pLines; pLines = t_plines_ini + 40;

        int min_rc = r < c ? r : c;
        for (int i = r - min_rc, j = c - min_rc;
             i < BOARD_SIZE && j < BOARD_SIZE;
             i ++, j ++) {
            lines[pLines++] = INT2CHAR(board[i * BOARD_SIZE + j]);
        }
        line_sizes[t_size_ini + 2] = pLines; pLines = t_plines_ini + 60;

        int dr = r < (BOARD_SIZE - 1 - c) ? r : (BOARD_SIZE - 1 - c);
        for (int i = r - dr, j = c + dr; i < BOARD_SIZE && j >= 0; i ++, j --) {
            lines[pLines++] = INT2CHAR(board[i * BOARD_SIZE + j]);
        }
        line_sizes[t_size_ini + 3] = pLines;
    }

    __syncthreads();

    // ================== calculate old scores ==================

    if (tid < 128) {
        for (int mov_id = 0; mov_id < num_next_loc; mov_id++) {
            tid_score[tid] = 0;

            int group_id = tid / 64;  // [0, 1]
            int group_tid = tid % 64; // [0, 63]

            char *pattern = patterns_p2 + 6 * (group_tid % 16);
            int *nxt = dfas_p2 + 7 * (group_tid % 16);

            if (group_id == 0 && player_num == 1 || group_id == 1 && player_num == 2) {
                pattern = patterns_p1 + 6 * (group_tid % 16);
                nxt = dfas_p1 + 7 * (group_tid % 16);
            }

            int i = 0, j;
            int m = pattern_size[(group_tid % 16)], n = line_size[mov_id * 4 + (group_tid / 16)];
            char *line = lines + 80 * mov_id + 20 * (group_tid / 16);

            while (i < n) {
                j = 0;
                while (i < n && j < m) {
                    if (j == -1 || line[i] == pattern[j]) {
                        i++;
                        j++;
                    } else {
                        j = nxt[j];
                    }
                }

                if (j == m) {
                    tid_score[tid] += score_map[group_tid % 16];
                    i = i - j + 1;
                    continue;
                }

                break;
            }

            __syncthreads();

            if (tid == 0) {
                int sum1 = 0, sum2 = 0;
                for (int t = 0; t < 64; t++) sum1 += tid_score[t];
                for (int t = 64; t < 128; t++) sum2 += tid_score[t];
                score[mov_id] = -(sum1 - sum2);
            }

            __syncthreads();
        }
    }

    __syncthreads();

    // ================== get all lines according to the moves (after placing it) ==================
    if (tid < num_next_loc) {
        int t_plines_ini = tid * 80;

        for (int i = t_plines_ini; i < t_plines_ini + 80; i++) {
            lines[i] = '0';
        }

        int t_size_ini = tid * 4;
        int pLines = t_plines_ini;
        int r = next_locations[tid] / BOARD_SIZE, c = next_locations[tid] % BOARD_SIZE;
        int local_board[BOARD_SIZE * BOARD_SIZE];

        for (int tmp_loc = 0; tmp_loc < BOARD_SIZE * BOARD_SIZE; tmp_loc ++) {
            local_board[tmp_loc] = board[tmp_loc];
        }

        if (max_layer == 1) { local_board[r * BOARD_SIZE + c] = player_num }
        else { local_board[r * BOARD_SIZE + c] = 3 - player_num }

        for (int i = 0; i < BOARD_SIZE; i++) {
            lines[pLines++] = INT2CHAR(local_board[i * BOARD_SIZE + c]);
        }
        line_sizes[t_size_ini] = pLines;
        pLines = t_plines_ini + 20;

        for (int j = 0; j < board_size; j++) {
            lines[pLines++] = INT2CHAR(local_board[r * BOARD_SIZE + j]);
        }
        line_sizes[t_size_ini + 1] = pLines;
        pLines = t_plines_ini + 40;

        int min_rc = r < c ? r : c;
        for (int i = r - min_rc, j = c - min_rc;
             i < BOARD_SIZE && j < BOARD_SIZE;
             i++, j++) {
            lines[pLines++] = INT2CHAR(local_board[i * BOARD_SIZE + j]);
        }
        line_sizes[t_size_ini + 2] = pLines;
        pLines = t_plines_ini + 60;

        int dr = r < (BOARD_SIZE - 1 - c) ? r : (BOARD_SIZE - 1 - c);
        for (int i = r - dr, j = c + dr; i < BOARD_SIZE && j >= 0; i++, j--) {
            lines[pLines++] = INT2CHAR(local_board[i * BOARD_SIZE + j]);
        }
        line_sizes[t_size_ini + 3] = pLines;
    }

    __syncthreads();

    // ================== calculate new scores ==================

    if (tid < 128) {
        for (int mov_id = 0; mov_id < num_next_loc; mov_id++) {
            tid_score[tid] = 0;

            int group_id = tid / 64;  // [0, 1]
            int group_tid = tid % 64; // [0, 63]

            char *pattern = patterns_p2 + 6 * (group_tid % 16);
            int *nxt = dfas_p2 + 7 * (group_tid % 16);

            if (group_id == 0 && player_num == 1 || group_id == 1 && player_num == 2) {
                pattern = patterns_p1 + 6 * (group_tid % 16);
                nxt = dfas_p1 + 7 * (group_tid % 16);
            }

            int i = 0, j;
            int m = pattern_size[(group_tid % 16)], n = line_size[mov_id * 4 + (group_tid / 16)];
            char *line = lines + 80 * mov_id + 20 * (group_tid / 16);

            while (i < n) {
                j = 0;
                while (i < n && j < m) {
                    if (j == -1 || line[i] == pattern[j]) {
                        i++;
                        j++;
                    } else {
                        j = nxt[j];
                    }
                }

                if (j == m) {
                    tid_score[tid] += score_map[group_tid % 16];
                    i = i - j + 1;
                    continue;
                }

                break;
            }

            __syncthreads();

            if (tid == 0) {
                int sum1 = 0, sum2 = 0;
                for (int t = 0; t < 64; t++) sum1 += tid_score[t];
                for (int t = 64; t < 128; t++) sum2 += tid_score[t];
                score[mov_id] += (sum1 - sum2);
            }

            __syncthreads();
        }
    }

    __syncthreads();

    if (tid == 0) {
        *moves_count = num_next_loc;
        for (int i = 0; i < num_next_loc; i ++) *res[i] = score[i];
    }
}

extern "C"
int heuristic_moves_cpu(int *res, int *next_locations, int *moves_count,
                        int* board, int player_num,
                        char* patterns_p1, char* patterns_p2,
                        int* dfas_p1, int* dfas_p2,
                        int* pattern_size, int* score_map,
                        int max_layer) {

    int *dev_res, *dev_moves_count, *dev_next_locations;
    int *dev_board;
    char *dev_patterns_p1, *dev_patterns_p2;
    int *dev_dfas_p1, *dev_dfas_p2;
    int *dev_pattern_size, *score_map;

    /* =============== Malloc memory on GPU =============== */
    checkCudaError(cudaMalloc((void**)&dev_res, sizeof(int) * 60), "Malloc res");
    checkCudaError(cudaMalloc((void**)&dev_next_locations, sizeof(int) * 60), "Malloc next locations");
    checkCudaError(cudaMalloc((void**)&dev_moves_count, sizeof(int)), "Malloc moves_count");
    checkCudaError(cudaMalloc((void**)&dev_board, sizeof(int) * BOARD_SIZE * BOARD_SIZE), "Malloc board");
    checkCudaError(cudaMalloc((void**)&dev_patterns_p1, sizeof(char) * 16 * 6), "Malloc patterns 1");
    checkCudaError(cudaMalloc((void**)&dev_patterns_p2, sizeof(char) * 16 * 6), "Malloc patterns 2");
    checkCudaError(cudaMalloc((void**)&dev_dfas_p1, sizeof(int) * 16 * 7), "Malloc dfa 1");
    checkCudaError(cudaMalloc((void**)&dev_dfas_p2, sizeof(int) * 16 * 7), "Malloc dfa 2");
    checkCudaError(cudaMalloc((void**)&dev_pattern_size, sizeof(int) * 16), "Malloc pattern size");
    checkCudaError(cudaMalloc((void**)&dev_score_map, sizeof(int) * 16), "Malloc score map");

    /* =============== Copy memory from RAM to GPU device =============== */
    checkCudaError(cudaMemcpy(dev_board, board, sizeof(int) * BOARD_SIZE * BOARD_SIZE, cudaMemcpyHostToDevice), "copy board");
    checkCudaError(cudaMemcpy(dev_patterns_p1, patterns_p1, sizeof(char) * 16 * 6, cudaMemcpyHostToDevice), "copy patterns 1");
    checkCudaError(cudaMemcpy(dev_patterns_p2, patterns_p2, sizeof(char) * 16 * 6, cudaMemcpyHostToDevice), "copy patterns 2");
    checkCudaError(cudaMemcpy(dev_dfas_p1, dfas_p1, sizeof(int) * 16 * 7, cudaMemcpyHostToDevice), "copy dfa 1");
    checkCudaError(cudaMemcpy(dev_dfas_p2, dfas_p2, sizeof(int) * 16 * 7, cudaMemcpyHostToDevice), "copy dfa 2");
    checkCudaError(cudaMemcpy(dev_pattern_size, pattern_size, sizeof(int) * 16, cudaMemcpyHostToDevice), "copy pattern size");
    checkCudaError(cudaMemcpy(dev_score_map, score_map, sizeof(int) * 16, cudaMemcpyHostToDevice), "copy score map");

    /* =============== Call kernel function =============== */
    heuristic_moves_kernel<<<1, BOARD_SIZE * BOARD_SIZE>>>(dev_res, dev_next_locations, dev_moves_count, dev_board, dev_patterns_p1,
                                                           dev_patterns_p2, dev_dfas_p1, dev_dfas_p2, dev_pattern_size,
                                                           dev_score_map, max_layer);

    checkCudaError(cudaMemcpy(res, dev_res, sizeof(int) * 60, cudaMemcpyDeviceToHost), "copy score from GPU");
    checkCudaError(cudaMemcpy(next_locations, dev_next_locations, sizeof(int) * 60, cudaMemcpyDeviceToHost), "copy locations from GPU");
    checkCudaError(cudaMemcpy(moves_count, dev_moves_count, sizeof(int), cudaMemcpyDeviceToHost), "copy count from GPU");

    /* =============== Free memory =============== */

    checkCudaError(cudaFree(dev_res), "free res");
    checkCudaError(cudaFree(dev_next_locations), "free moves_count");
    checkCudaError(cudaFree(dev_moves_count), "free moves_count");
    checkCudaError(cudaFree(dev_board), "free board");
    checkCudaError(cudaFree(dev_patterns_p1), "free pattern 1");
    checkCudaError(cudaFree(dev_patterns_p2), "free pattern 1");
    checkCudaError(cudaFree(dev_dfas_p1), "free dfa 1");
    checkCudaError(cudaFree(dev_dfas_p2), "free dfa 2");
    checkCudaError(cudaFree(dev_pattern_size), "free pattern size");
    checkCudaError(cudaFree(dev_score_map), "free score map");

    return 0;
}