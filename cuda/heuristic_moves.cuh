#ifndef GOMOKU_HEURISTIC_MOVES_H
#define GOMOKU_HEURISTIC_MOVES_H

extern "C"
int heuristic_moves_cpu(int *res, int *next_locations, int *moves_count,
                        int* board, int player_num,
                        char* patterns_p1, char* patterns_p2,
                        int* dfas_p1, int* dfas_p2,
                        int* pattern_size, int* score_map,
                        int max_layer);


#endif //GOMOKU_HEURISTIC_MOVES_H