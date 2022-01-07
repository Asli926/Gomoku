#include "../include/pattern_search.h"

int test_search() {
    std::string pattern{"0001100"};
    std::string line{"00001100011000"};
    std::vector<int> nxt = construct_nxt(pattern);
    int res = match_count(line, pattern, nxt);
    printf("match count: %d\n", res);
}