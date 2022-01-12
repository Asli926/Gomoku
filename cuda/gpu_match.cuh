#ifndef GOMOKU_GPU_MATCH_H
#define GOMOKU_GPU_MATCH_H

extern "C"
int match_count_multiple(char* lines, int* line_size, int player_num);

extern "C"
int setPatternRelatedInfo(char* patterns_p1_, char* patterns_p2_, int* dfas_p1_, int* dfas_p2_,
                          int* pattern_size_, int* score_map_);


#endif //GOMOKU_GPU_MATCH_H