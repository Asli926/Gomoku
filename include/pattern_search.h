#ifndef GOMOKU_PATTERN_SEARCH_H
#define GOMOKU_PATTERN_SEARCH_H

#include <string>
#include <vector>

std::vector<int> construct_nxt(const std::string& pattern);
int match_count(const std::string& line, const std::string& pattern, const std::vector<int>& nxt);

#endif //GOMOKU_PATTERN_SEARCH_H
