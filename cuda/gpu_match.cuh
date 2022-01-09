#ifndef GOMOKU_GPU_MATCH_H
#define GOMOKU_GPU_MATCH_H

extern "C"
int match_count_multiple(char* lines, char* patterns, int* dfas, int* pattern_size, int* line_size, int* score_map);


#endif //GOMOKU_GPU_MATCH_H