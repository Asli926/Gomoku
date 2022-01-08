#include "../include/pattern_search.h"

int* construct_nxt(const std::string& pattern) {
    int m = pattern.size();
    int i = 1, j = 0;
    int* nxt = new int[m + 1];
    nxt[0] = -1;

    for (int k = 1; k < m + 1; k ++) nxt[k] = 0;

    while (i < m) {
        if (j == -1 || pattern[i] == pattern[j]) {
            i ++; j ++;
            nxt[i] = j;
        } else {
            j = nxt[j];
        }
    }

    return nxt;
}

int match_count(const std::string& line, const std::string& pattern, const int* nxt) {
    int i = 0, j;
    int m = pattern.size(), n = line.size();
    int res = 0;

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
            res ++;
            i = i - j + 1;
            continue;
        }

        // Otherwise: we have traversed to the end of 'line', so just break
        break;
    }

    return res;
}
