#ifndef join_FOR_CPP_2675DCD0_9480_4c0c_B92A_CC14C027B731
#define join_FOR_CPP_2675DCD0_9480_4c0c_B92A_CC14C027B731

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <string>
#include <vector>

#include "trie.h"
#include "utf8.h"
#include "util.h"
#include "wSED.h"

#define PRIME 2147483647  // (1 << 31) - 1
#define TOPK_HASH_SIZE 10000
#define TASTE_HASH_SIZE 10000
// #define JOIN_INFO
// #define JOIN_DEBUG
#define SODDY_PACK

using namespace std;
using namespace wSED;
using namespace chrono;
using namespace util;

typedef tuple<int, int> TI2;
typedef tuple<int, int, int> TI3;
typedef TrieStore<char32_t, vector<int> *> TriePart;
static int _n_prfx_;

namespace join {
string args_str;

class CustomStringHash {
public:
    size_t operator()(const u32string &k) const {
        size_t hash = 0;
        for (auto ch : k) {
            hash += std::hash<char32_t>{}(ch);
            // hash += ch;
        }
        return hash;
    }
};

vector<u32string> distinct_prefix(vector<u32string> &S_Q) {
    vector<u32string> S_P;
    unordered_set<u32string> prefix_set;
    for (auto q : S_Q) {
        for (int i = 1; i <= int(q.size()); ++i) {
            auto p = q.substr(0, i);
            if (prefix_set.find(p) == prefix_set.end()) {
                prefix_set.insert(p);
                S_P.push_back(p);
            }
        }
    }
    return S_P;
}

CT count_table(vector<u32string> &S_Q, vector<u32string> &S_D, int delta_M, bool prefix_aug) {
    vector<u32string> S_P;
    vector<u32string> *S_QP = &S_Q;
    if (prefix_aug) {
        S_P = distinct_prefix(S_Q);
        S_QP = &S_P;
    }

    CT ct;
    int delta;
    for (auto q : *S_QP) {
        for (auto s : S_D) {
            auto dist = SED(q, s);
            for (delta = dist; delta <= delta_M; ++delta) {
                auto qry = Qry(q, delta);
                if (ct.find(qry) == ct.end()) {
                    ct[qry] = 0;
                }
                ct[qry] += 1;
            }
        }
    }

    return ct;
}

vector<vector<int>> all_pair_join(vector<u32string> &S_Q, vector<u32string> &S_D, int delta_M) {
// naive algorithm by computing pairwise distance computation
#ifdef JOIN_INFO
    system_clock::time_point tps;
    system_clock::time_point tpe;
    duration<double> trie_build_time = duration<double>::zero();
    duration<double> filter_time = duration<double>::zero();
    duration<double> cal_dist_time = duration<double>::zero();
    duration<double> total_time = duration<double>::zero();
    long int total_line = 0;
    long int compute_line = 0;
    long int prune_line = 0;
    long int total_cell = 0;
    long int compute_cell = 0;
    long int prune_cell = 0;
#endif
    // allocate & initialize memories for dynamic programming
    vector<vector<int>> output;
    for (int i = 0; i < (int)S_Q.size(); i++) {
        vector<int> res(delta_M + 1, 0);
        output.push_back(res);
    }
    u32string r, s;

#ifndef JOIN_INFO
    int sed;
    for (int rid = 0; rid < (int)S_Q.size(); rid++) {
        r = S_Q[rid];
        for (int sid = 0; sid < (int)S_D.size(); sid++) {
            s = S_D[sid];
            sed = wSED::SED(r, s);
            if (sed <= delta_M) {
                output[rid][sed] += 1;
            }
        }
    }

    for (int rid = 0; rid < (int)S_Q.size(); rid++) {  // accumlation make slightly faster
        for (int i = 1; i < delta_M + 1; i++) {
            output[rid][i] += output[rid][i - 1];
        }
    }
#endif
#ifdef JOIN_INFO
    vector<int> r_lens;
    for (auto r : S_Q) {
        r_lens.push_back(r.length());
    }
    vector<int> s_lens;
    for (auto s : S_D) {
        s_lens.push_back(s.length());
    }

    int R_max = *max_element(begin(r_lens), end(r_lens));
    // int S_max = *max_element(begin(s_lens), end(s_lens));

    long int r_total = accumulate(r_lens.begin(), r_lens.end(), 0);
    long int s_total = accumulate(s_lens.begin(), s_lens.end(), 0);

    total_line = r_total * S_D.size();
    compute_line = total_line;
    prune_line = total_line - compute_line;

    total_cell = r_total * s_total;
    compute_cell = total_cell;
    prune_cell = total_cell - compute_cell;
    printf("r_max        : %15d\n", R_max);
    printf("r_total      : %15ld\n", r_total);
    printf("s_total      : %15ld\n", s_total);
    printf("Total line   : %15ld\n", total_line);
    printf("Computed line: %15ld\n", compute_line);
    printf("Pruned line  : %15ld\n", prune_line);
    printf("Total cell   : %15ld\n", total_cell);
    printf("Computed cell: %15ld\n", compute_cell);
    printf("Pruned cell  : %15ld\n", prune_cell);

    string statFileName = "stat/" + args_str + ".txt";
    auto writeFile = fopen(statFileName.c_str(), "w");
    fprintf(writeFile, "[%s] %s\n",
            getTimeStamp().c_str(), args_str.c_str());
    fprintf(writeFile, "Total line   : %15ld\n", total_line);
    fprintf(writeFile, "Computed line: %15ld\n", compute_line);
    fprintf(writeFile, "Pruned line  : %15ld\n", prune_line);
    fprintf(writeFile, "Total cell   : %15ld\n", total_cell);
    fprintf(writeFile, "Computed cell: %15ld\n", compute_cell);
    fprintf(writeFile, "Pruned cell  : %15ld\n", prune_cell);
    fclose(writeFile);
#endif
    return output;
}

vector<vector<int>> Ablation_join(vector<u32string> &R, vector<u32string> &S, int delta) {
#ifdef JOIN_INFO
    system_clock::time_point tps;
    system_clock::time_point tpe;
    duration<double> duration_time = duration<double>::zero();
    long int total_line = 0;
    long int share_line = 0;
    long int compute_line = 0;
    long int prune_line = 0;
    long int total_cell = 0;
    long int share_cell = 0;
    long int compute_cell = 0;
    long int prune_cell = 0;
#endif

    int n_qry = (int)R.size();

    u32string s;
    u32string r;

    vector<int> r_lens;
    vector<int> s_lens;
    int R_max = 0;
    int S_max = 0;
    for (auto r : R) {
        r_lens.push_back(r.size());
    }

    for (auto s : S) {
        s_lens.push_back(s.size());
    }
    R_max = *max_element(begin(r_lens), end(r_lens));
    S_max = *max_element(begin(s_lens), end(s_lens));

    // initialize join_result
    vector<vector<int>> join_result;
    for (int i = 0; i < (int)n_qry; i++) {
        vector<int> res(delta + 1, 0);
        join_result.push_back(res);
    }

    // allocate & initialize memories for dynamic programming
    int **mem;
    mem = new int *[R_max + 1];
    for (int i = 0; i < R_max + 1; ++i) {
        mem[i] = new int[S_max + 2];
        mem[i][0] = i;
        for (int j = 1; j < S_max + 2; ++j) {
            mem[i][j] = 0;
        }
    }

    int r_len;
    int s_len;
    int js;
    int je;
    int js_n;
    int je_n;
    int sed;

    // start iterating database strings
    // cout << "start iterate database strings" << endl;
    for (int sid = 0; sid < (int)S.size(); sid++) {
        // fetching
        s = S[sid];
        s_len = s.size();

        for (int rid = 0; rid < (int)R.size(); rid++) {
            r = R[rid];
            r_len = r.size();
            js = 1;
            je = s_len + 1;
            sed = INT32_MAX;
            for (int i = 1; i < r_len + 1; ++i) {
#ifdef JOIN_INFO
                if (je > js) {
                    compute_line += 1;
                }
#endif
                for (int j = js; j < je; ++j) {
                    if (r[i - 1] == s[j - 1]) {  // i and j are 1-index
                        mem[i][j] = mem[i - 1][j - 1];
                    } else {
                        // mem[i][j] = min({mem[i - 1][j - 1], mem[i][j - 1], mem[i - 1][j]}) + 1;
                        mem[i][j] = MIN(mem[i - 1][j], mem[i - 1][j - 1]);  // faster than min funtion
                        mem[i][j] = MIN(mem[i][j], mem[i][j - 1]) + 1;      // faster than min funtion
                    }
#ifdef JOIN_INFO
                    compute_cell += 1;
#endif
                }

                if (i == r_len) {
                    sed = util::minArr(mem[r_len], js, je);
                    break;
                }

                // update next interval
                if (i > delta + 1) {
                    js_n = INT32_MAX;
                    je_n = 0;
                    for (int j = js; j < MIN(je, s_len); ++j) {
                        if (mem[i][j] <= delta) {
                            js_n = j + 1;  // inclusive
                            break;
                        }
                    }
                    if (js_n < INT32_MAX) {
                        for (int j = MIN(je, s_len) - 1; j >= js; --j) {
                            if (mem[i][j] <= delta) {
                                je_n = j + 2;  // exclusive
                                break;
                            }
                        }
                    }
                    if (js_n > je_n) {
                        break;
                    }

                    // initialize boundary
                    if (je == je_n - 1) {
                        mem[i][je_n - 1] = INT32_MAX;
                    }
                    if (js_n > 1) {
                        mem[i + 1][js_n - 1] = INT32_MAX;
                    }
                    js = js_n;
                    je = je_n;
                }
            }
            if (sed <= delta) {
                join_result[rid][sed] += 1;
            }
        }
    }

    for (int rid = 0; rid < (int)n_qry; rid++) {  // accumlation make slightly faster
        for (int i = 1; i < delta + 1; i++) {
            join_result[rid][i] += join_result[rid][i - 1];
        }
    }

#ifdef JOIN_INFO
    long int r_total = accumulate(r_lens.begin(), r_lens.end(), 0);
    long int s_total = accumulate(s_lens.begin(), s_lens.end(), 0);
    // long int n_prfx = trie->size() - 1;

    total_line = r_total * S.size();
    // share_line = n_prfx * S.size();
    prune_line = total_line - compute_line;

    total_cell = r_total * s_total;
    // share_cell = n_prfx * s_total;
    prune_cell = total_cell - compute_cell;
    printf("r_max        : %15d\n", R_max);
    printf("r_total      : %15ld\n", r_total);
    printf("s_total      : %15ld\n", s_total);
    // printf("n_prfx       : %15ld\n", n_prfx);
    printf("Total line   : %15ld\n", total_line);
    printf("Shared line  : %15ld\n", share_line);
    printf("Computed line: %15ld\n", compute_line);
    printf("Pruned line  : %15ld\n", prune_line);
    printf("Total cell   : %15ld\n", total_cell);
    printf("Shared cell  : %15ld\n", share_cell);
    printf("Computed cell: %15ld\n", compute_cell);
    printf("Pruned cell  : %15ld\n", prune_cell);

    string statFileName = "stat/" + args_str + ".txt";
    auto writeFile = fopen(statFileName.c_str(), "w");
    fprintf(writeFile, "[%s] %s\n",
            getTimeStamp().c_str(), args_str.c_str());
    fprintf(writeFile, "Total line   : %15ld\n", total_line);
    fprintf(writeFile, "Computed line: %15ld\n", compute_line);
    fprintf(writeFile, "Pruned line  : %15ld\n", prune_line);
    fprintf(writeFile, "Total cell   : %15ld\n", total_cell);
    fprintf(writeFile, "Computed cell: %15ld\n", compute_cell);
    fprintf(writeFile, "Pruned cell  : %15ld\n", prune_cell);
    fclose(writeFile);
#endif
    for (int i = 0; i < R_max + 1; ++i) {
        delete[] mem[i];
    }
    delete[] mem;
    return join_result;
}

vector<vector<int>> SODDY_join(int R_size, vector<u32string> &S, int delta, int prune_level, bool prefix_mode = false, char32_t **S_Q_ptr = nullptr) {
// prune_level0: sharing
// prune_level1: sharing + prefix pruning
// prune_level2: sharing + interval pruning
#ifdef JOIN_INFO
    system_clock::time_point tps;
    system_clock::time_point tpe;
    duration<double> duration_time = duration<double>::zero();
    long int total_line = 0;
    long int share_line = 0;
    long int compute_line = 0;
    long int prune_line = 0;
    long int total_cell = 0;
    long int share_cell = 0;
    long int compute_cell = 0;
    long int prune_cell = 0;
#endif

    int n_qry = 0;

    u32string s;
    // u32string r;
    char32_t *r;
    int r_len;

    // start computing prefix
    // u32string prev_r = U"";
    char32_t *prev_r = nullptr;
    int prev_r_size = 0;
    int min_r_prev_r_size;

    // memorize previous prefix position
    vector<int> prfx_pos(R_size);  // last pos of common prefix in case 1-index <=> # of characters in common prefix
    vector<int> cum_distinct_prfx_offset(R_size);
    int k = 0;

    for (int rid = 0; rid < R_size; rid++) {
        r = S_Q_ptr[rid] + 1;
        r_len = *S_Q_ptr[rid];
        min_r_prev_r_size = MIN(r_len, prev_r_size);
        for (k = 0; k < min_r_prev_r_size; ++k) {
            if (r[k] != prev_r[k]) {
                break;
            }
        }
        prfx_pos[rid] = k;
        prev_r = r;
        prev_r_size = r_len;
        if (prefix_mode) {
            cum_distinct_prfx_offset[rid] = (n_qry - k - 1);
            n_qry += (r_len - k);
        }
    }

    if (!prefix_mode) {
        n_qry = R_size;
    }
#ifdef JOIN_DEBUG
    auto outFile = fopen("debug_0921.txt", "w");
    if (prefix_mode) {
        auto trie = new Trie<char32_t>();
        auto n_qry_test = trie->add_all_prefixes(R);
        assert(n_qry == n_qry_test);
    }
#endif

    vector<int> r_lens;
    for (int i = 0; i < R_size; ++i) {
        r_lens.push_back(*S_Q_ptr[i]);
    }

    vector<int> s_lens;
    for (auto s : S) {
        s_lens.push_back(s.size());
    }
    // cout << "r_lens: ";
    // util::printVec(r_lens);
    // cout << "s_lens: ";
    // util::printVec(s_lens);

    int R_max = *max_element(begin(r_lens), end(r_lens));
    int S_max = *max_element(begin(s_lens), end(s_lens));
    // cout<<"R, S " << R_max << ", " << S_max << endl;

    // initialize join_result
    vector<vector<int>> join_result;
    for (int i = 0; i < (int)n_qry; i++) {
        vector<int> res(delta + 1, 0);
        join_result.push_back(res);
    }

    // allocate & initialize memories for dynamic programming
    int **mem = new int *[R_max + 1];
    for (int i = 0; i < R_max + 1; ++i) {
        mem[i] = new int[S_max + 2];
        mem[i][0] = i;
        for (int j = 1; j < S_max + 2; ++j) {
            mem[i][j] = 0;
        }
    }

    int pruned_pos = 0;  // prune_level >= 1

    vector<int> req_js(R_max + 2);  // prune_level >= 2
    vector<int> req_je(R_max + 2);

    int s_len;
    int js = 1;  // inclusive
    int je;      // exclusive
    int js_n;    // inclusive
    int je_n;    // exclusive
    int sed = 0;
    int n_line_computed = 0;
    int last_point;
    int rid_prime;

    // start iterating database strings
    // cout << "start iterate database strings" << endl;
    for (int sid = 0; sid < (int)S.size(); sid++) {
        // fetching
        s = S[sid];
        s_len = int(s.size());
        je = s_len + 1;

        // initialize previous pruned position
        if (prune_level >= 1) {
            // in case when prune_pos < infinity, the following conditions hold.
            // (1) sed(prev_r[:pruned_pos], s) > delta
            // (2) when prune_level == 2, sed(prev_r[:(pruned_pos-1)], s) <= delta
            pruned_pos = INT32_MAX;
        }

        // initialize required interval
        if (prune_level >= 2) {
            req_js[1] = 1;
            req_je[1] = s_len + 1;
        }

        // start iterating query strings
        // cout << "start iterate query strings" << endl;
        for (int rid = 0; rid < R_size; rid++) {
            r = S_Q_ptr[rid] + 1;
            r_len = *S_Q_ptr[rid];

            // prefix pruning
            if (prune_level >= 1) {                 // should use prefix pruning even in case prune_level 2 for sort based algorithm
                if (pruned_pos <= prfx_pos[rid]) {  // (1) sed(prev_r[:pruned_pos], s) > delta
                    continue;
                } else {
                    // update n_line_computed
                    if (rid < R_size) {
                        // n_line_computed = prfx_pos[rid + 1];
                        n_line_computed = MIN(n_line_computed, prfx_pos[rid]);
                    }
                }
            }

            if (prune_level <= 2) {
                n_line_computed = prfx_pos[rid];
            }

            for (int i = n_line_computed + 1; i < r_len + 1; ++i) {
                // cout << "i: " << i << " curr candidate: [" << req_intvals[i][0] << ", " << req_intvals[i][1] << ")" << endl;
                if (prune_level >= 2) {
                    js = req_js[i];
                    je = req_je[i];
                    if (js >= je) {
                        pruned_pos = i;
                        break;
                    }
                    // assert(js <= s_len);
                    // assert(je <= s_len + 1);
                    mem[i][js - 1] = i;
                    mem[i][je] = i;
                }

                for (int j = js; j < je; ++j) {
                    // cout << "i, j: " << i << ", " << j << endl;
                    // assert(i > 0 && i < R_max + 1);
                    // assert(j > 0 && j < S_max + 1);
                    // assert(i - 1 < int(r.size()) && i - 1 >= 0);
                    // assert(j - 1 < int(s.size()) && j - 1 >= 0);
                    if (r[i - 1] == s[j - 1]) {  // i and j are 1-index
                        mem[i][j] = mem[i - 1][j - 1];
                    } else {
                        // mem[i][j] = min({mem[i - 1][j - 1], mem[i][j - 1], mem[i - 1][j]}) + 1;
                        mem[i][j] = MIN(mem[i - 1][j], mem[i - 1][j - 1]);  // faster than min funtion
                        mem[i][j] = MIN(mem[i][j], mem[i][j - 1]) + 1;      // faster than min funtion
                    }
                }
#ifdef JOIN_INFO
                if (je > js) {
                    compute_line += 1;
                    compute_cell += (je - js);
                }
#endif
                n_line_computed += 1;

                // compute pruned postion
                if (prune_level == 1) {
                    sed = util::minArr(mem[i], js, je);
                    if (sed > delta) {
                        pruned_pos = i;
                        break;
                    }
                }

                // compute the next required interval
                if (prune_level >= 2) {
                    if (i <= delta) {
                        js_n = 1;
                        je_n = s_len + 1;
                    } else {
                        js_n = INT32_MAX;
                        je_n = 0;
                        last_point = MIN(je, s_len);
                        for (int j = js; j < last_point; ++j) {
                            if (mem[i][j] <= delta) {
                                js_n = j + 1;  // inclusive
                                break;
                            }
                        }
                        if (js_n < INT32_MAX) {
                            for (int j = last_point - 1; j >= js; --j) {
                                if (mem[i][j] <= delta) {
                                    je_n = j + 2;  // exclusive
                                    break;
                                }
                            }
                        }
                    }
                    req_js[i + 1] = js_n;
                    req_je[i + 1] = je_n;

                    if (req_js[i + 1] >= req_je[i + 1]) {
                        pruned_pos = i;
                    }
                }

                if (prefix_mode || i == (int)r_len) {
                    if (prefix_mode) {
                        rid_prime = cum_distinct_prfx_offset[rid] + i;
                    } else {
                        rid_prime = rid;
                    }
                    if (prune_level != 1) {
                        sed = util::minArr(mem[i], js, je);
                    }
                    if (sed <= delta) {
                        join_result[rid_prime][sed] += 1;
                    }

                    if (prune_level >= 1) {
                        // printf("rid: %d sed: %d\n", rid, sed);
                        if (sed <= delta) {
                            pruned_pos = INT32_MAX;
                        } else {
                            pruned_pos = r_len;
                        }
                    }
                }
            }
        }
    }

    for (int rid = 0; rid < (int)n_qry; rid++) {  // accumlation make slightly faster
        for (int i = 1; i < delta + 1; i++) {
            join_result[rid][i] += join_result[rid][i - 1];
        }
    }
#ifdef JOIN_INFO
    long int r_total = accumulate(r_lens.begin(), r_lens.end(), 0);
    long int s_total = accumulate(s_lens.begin(), s_lens.end(), 0);
    long int n_prfx = _n_prfx_;

    total_line = r_total * S.size();
    share_line = n_prfx * S.size();
    prune_line = total_line - compute_line;

    total_cell = r_total * s_total;
    share_cell = n_prfx * s_total;
    prune_cell = total_cell - compute_cell;
    printf("r_max        : %15d\n", R_max);
    printf("r_total      : %15ld\n", r_total);
    printf("s_total      : %15ld\n", s_total);
    printf("n_prfx       : %15ld\n", n_prfx);
    printf("Total line   : %15ld\n", total_line);
    printf("Shared line  : %15ld\n", share_line);
    printf("Computed line: %15ld\n", compute_line);
    printf("Pruned line  : %15ld\n", prune_line);
    printf("Total cell   : %15ld\n", total_cell);
    printf("Shared cell  : %15ld\n", share_cell);
    printf("Computed cell: %15ld\n", compute_cell);
    printf("Pruned cell  : %15ld\n", prune_cell);

    string statFileName = "stat/" + args_str + ".txt";
    auto writeFile = fopen(statFileName.c_str(), "w");
    fprintf(writeFile, "[%s] %s\n",
            getTimeStamp().c_str(), args_str.c_str());
    fprintf(writeFile, "Total line   : %15ld\n", total_line);
    fprintf(writeFile, "Computed line: %15ld\n", compute_line);
    fprintf(writeFile, "Pruned line  : %15ld\n", prune_line);
    fprintf(writeFile, "Total cell   : %15ld\n", total_cell);
    fprintf(writeFile, "Computed cell: %15ld\n", compute_cell);
    fprintf(writeFile, "Pruned cell  : %15ld\n", prune_cell);
    fclose(writeFile);
#endif

    for (int i = 0; i < R_max + 1; ++i) {
        delete[] mem[i];
    }
    delete[] mem;
    return join_result;
}

vector<vector<int>> TEDDY_join(vector<u32string> &R, vector<u32string> &S, int delta, int prune_level, bool prefix_mode = false) {
// prune_level0: sharing
// prune_level1: sharing + prefix pruning
// prune_level2: sharing + interval pruning
#ifdef JOIN_INFO
    system_clock::time_point tps;
    system_clock::time_point tpe;
    duration<double> duration_time;
    tps = system_clock::now();
    long int total_line = 0;
    long int share_line = 0;
    long int compute_line = 0;
    long int prune_line = 0;
    long int total_cell = 0;
    long int share_cell = 0;
    long int compute_cell = 0;
    long int prune_cell = 0;
#endif
    int n_qry;
    Trie<char32_t> *trie_src = new Trie<char32_t>(_n_prfx_);

    if (prefix_mode) {
        n_qry = trie_src->add_all_prefixes(R);
    } else {
        trie_src->add_strings(R);
        n_qry = (int)R.size();
    }

    RadixTree<char32_t> *trie = new RadixTree<char32_t>(trie_src);
    trie->root->set_next();
    // trie_src->root->print_tree();
    // cout << endl;
    delete trie_src;
    // for (int i = 0; i < _n_prfx_ + 1; ++i) {
    //     cout << trie->rid_arr[i] << " ";
    // }
    // cout << endl;
    // for (int i = 0; i < _n_prfx_ + 1; ++i) {
    //     wcout << (wchar_t)trie->chr_arr[i] << " ";
    // }
    // cout << endl;
    // trie->root->print_tree();
    // RadixTreeNode<> *traverse_node = trie->root->children[0];
    // while (traverse_node) {
    //     for (int i = 0; i < traverse_node->n_str; ++i) {
    //         wcout << (wchar_t) * (traverse_node->s_str + i) << " ";
    //     }
    //     cout << endl;
    //     if (traverse_node->is_leaf()) {
    //         traverse_node = traverse_node->next_node;
    //     } else {
    //         traverse_node = traverse_node->children[0];
    //     }
    // }

#ifdef JOIN_INFO
    // cout << "single edge: " << trie->root->single_edge_count() << endl;
    // cout << "all edge   : " << trie->root->size() - 1 << endl;
    // cout << "all queries: " << R.size() << endl;
    tpe = system_clock::now();
    duration_time = tpe - tps;
    printf("build trie time: %fs \n", duration_time.count());
    // printf("build trie time: %fs \n", duration_time);
#endif

    vector<int> r_lens;
    for (auto r : R) {
        r_lens.push_back(r.length());
    }
    vector<int> s_lens;
    for (auto s : S) {
        s_lens.push_back(s.length());
    }
    int R_max = *max_element(begin(r_lens), end(r_lens));
    int S_max = *max_element(begin(s_lens), end(s_lens));
    // cout<<"R, S " << R_max << ", " << S_max << endl;

    // initialize join_result
    vector<vector<int>> join_result;
    for (int i = 0; i < (int)n_qry; i++) {
        vector<int> res(delta + 1, 0);
        join_result.push_back(res);
    }

    // allocate & initialize memories for dynamic programming
    int **mem = new int *[R_max + 1];
    for (int i = 0; i < R_max + 1; i++) {
        mem[i] = new int[S_max + 2];
        for (int j = 1; j < S_max + 2; j++) {
            mem[i][j] = 0;
        }
        mem[i][0] = i;
    }

    u32string s;
    // u32string r;

    vector<int> req_js(R_max + 2);  // prune_level >= 2
    vector<int> req_je(R_max + 2);

    if (prune_level >= 2) {
        for (int i = 1; i < delta + 2; i++) {
            req_js[i] = 1;
        }
    }

    int s_len;
    int js = 1;  // inclusive
    int je;      // exclusive
    int js_n;    // inclusive
    int je_n;    // exclusive
    int sed;
    int i;
    int rid;
    int len_node;
    int last_point;
    char32_t *rc_ptr;
    char32_t rc;
    // char32_t sc;

    // vector<TrieNode<> *> node_stack;
    // RadixTreeNode<> *node_stack[10000];
    // RadixTreeNode<> **node_stack_ptr = node_stack;

    // start iterating database strings
    // cout << "start iterate database strings" << endl;
    RadixTreeNode<char32_t> *curr_node = nullptr;
    // RadixTreeNode<char32_t> *child_node;
    for (int sid = 0; sid < (int)S.size(); sid++) {
        // fetching
        s = S[sid];
        s_len = int(s.size());
        je = s_len + 1;

        // initialize required interval
        if (prune_level >= 2) {
            for (int i = 1; i < delta + 2; i++) {
                req_je[i] = s_len + 1;
            }
        }

        // start iterating query strings
        // node_stack.clear();

        if (!trie->root->is_leaf()) {
            curr_node = trie->root->children[0];
        }
        // curr_node = trie->root->children;
        // if (!curr_node->is_leaf()) {
        //     child_node = curr_node->children;
        //     for (int cid = 0; cid < curr_node->n_child; ++cid) {
        //         // *node_stack_ptr = curr_node->children[cid];
        //         // *node_stack_ptr = curr_node->children + cid;
        //         *node_stack_ptr = child_node;
        //         child_node = child_node->sibling;
        //         ++node_stack_ptr;
        //     }
        // }

        // while (node_stack.size() > 0) {
        // while (node_stack_cnt > 0) {
        while (curr_node) {
            // curr_node = node_stack.back();
            // node_stack.pop_back();
            // node_stack_cnt -= 1;
            // curr_node = node_stack[--node_stack_cnt];
            // --node_stack_ptr;
            // curr_node = *node_stack_ptr;

            i = curr_node->depth - 1;
            len_node = curr_node->n_str;
            rc_ptr = curr_node->s_str;
            for (int pos = 0; pos < len_node; ++pos) {
                i += 1;  // curr_node-> depth + pos
                rc = *rc_ptr;
                rc_ptr += 1;
                // rc = curr_node->chr(pos);
                // rc = *(curr_node->s_str + pos);
                if (prune_level >= 2) {
                    js = req_js[i];
                    je = req_je[i];
                    if (i > delta) {
                        mem[i][js - 1] = i;
                        mem[i][je] = i;
                    }
                }
#ifdef JOIN_INFO
                if (je > js) {
                    compute_line += 1;
                }
#endif
                for (int j = js; j < je; j++) {
                    // sc = s[j - 1];
                    if (rc == s[j - 1]) {
                        mem[i][j] = mem[i - 1][j - 1];
                    } else {
                        // mem[p][q] = min({mem[p - 1][q], mem[p][q - 1], mem[p - 1][q - 1]}) + 1;
                        mem[i][j] = MIN(mem[i - 1][j], mem[i - 1][j - 1]);  // faster than min funtion
                        mem[i][j] = MIN(mem[i][j], mem[i][j - 1]) + 1;      // faster than min funtion
                    }
#ifdef JOIN_INFO
                    compute_cell += 1;
#endif
                }
                if (prune_level == 1) {
                    sed = util::minArr(mem[i], js, je);
                    if (sed > delta) {
                        break;
                    }
                }

                if (prune_level >= 2 && i > delta) {
                    js_n = INT32_MAX;
                    je_n = 0;
                    last_point = MIN(je, s_len);
                    for (int j = js; j < last_point; ++j) {
                        if (mem[i][j] <= delta) {
                            js_n = j + 1;  // inclusive
                            break;
                        }
                    }
                    if (js_n < INT32_MAX) {
                        for (int j = last_point - 1; j >= js; --j) {
                            if (mem[i][j] <= delta) {
                                je_n = j + 2;  // exclusive
                                break;
                            }
                        }
                    }
                    req_js[i + 1] = js_n;
                    req_je[i + 1] = je_n;
                }

                // rid = *(curr_node->s_rid + pos);
                rid = curr_node->rid(pos);
                // cout << "rid: " << rid << endl;
                // if (curr_node->is_active(pos)) {
                if (rid >= 0) {
                    if (prune_level != 1) {
                        sed = util::minArr(mem[i], js, je);
                    }
                    // sed = util::minArr(mem[i], js, je);
                    if (sed <= delta) {
                        // join_result[curr_node->rid(pos)][sed] += 1;
                        join_result[rid][sed] += 1;
                    }
                }
                if (!(prune_level < 2 || req_js[i + 1] < req_je[i + 1])) {
                    break;
                }
            }

            if (!curr_node->is_leaf() && (prune_level < 2 || req_js[i + 1] < req_je[i + 1])) {
                // child_node = curr_node->children;
                curr_node = curr_node->children[0];
                // for (int cid = 0; cid < curr_node->n_child; ++cid) {
                //     // *node_stack_ptr = curr_node->children[cid];
                //     // *node_stack_ptr = curr_node->children + cid;
                //     *node_stack_ptr = child_node;
                //     child_node = child_node->sibling;
                //     ++node_stack_ptr;
                // }
            } else {
                curr_node = curr_node->next_node;
            }
        }

#ifdef JOIN_INFO
        if (sid % 100 == 0) {
            cout << "\rS " << sid << "th done" << flush;
        }
#endif
    }
#ifdef JOIN_INFO
    cout << endl;
#endif

    for (int rid = 0; rid < (int)n_qry; rid++) {  // accumlation make slightly faster
        for (int i = 1; i < delta + 1; i++) {
            join_result[rid][i] += join_result[rid][i - 1];
        }
    }
#ifdef JOIN_INFO
    long int r_total = accumulate(r_lens.begin(), r_lens.end(), 0);
    vector<u32string> S_P = distinct_prefix(R);
    vector<int> p_lens;
    for (auto q : S_P) {
        p_lens.push_back(q.length());
    }
    if (prefix_mode) {
        r_total = accumulate(p_lens.begin(), p_lens.end(), 0);
    }
    long int s_total = accumulate(s_lens.begin(), s_lens.end(), 0);
    long int n_prfx = (int)S_P.size();

    total_line = r_total * S.size();
    share_line = n_prfx * S.size();
    prune_line = total_line - compute_line;

    total_cell = r_total * s_total;
    share_cell = n_prfx * s_total;
    prune_cell = total_cell - compute_cell;
    printf("r_max        : %15d\n", R_max);
    printf("r_total      : %15ld\n", r_total);
    printf("s_total      : %15ld\n", s_total);
    printf("n_prfx       : %15ld\n", n_prfx);
    printf("Total line   : %15ld\n", total_line);
    printf("Shared line  : %15ld\n", share_line);
    printf("Computed line: %15ld\n", compute_line);
    printf("Pruned line  : %15ld\n", prune_line);
    printf("Total cell   : %15ld\n", total_cell);
    printf("Shared cell  : %15ld\n", share_cell);
    printf("Computed cell: %15ld\n", compute_cell);
    printf("Pruned cell  : %15ld\n", prune_cell);

    string statFileName = "stat/" + args_str + ".txt";
    auto writeFile = fopen(statFileName.c_str(), "w");
    fprintf(writeFile, "[%s] %s\n",
            getTimeStamp().c_str(), args_str.c_str());
    fprintf(writeFile, "Total line   : %15ld\n", total_line);
    fprintf(writeFile, "Computed line: %15ld\n", compute_line);
    fprintf(writeFile, "Pruned line  : %15ld\n", prune_line);
    fprintf(writeFile, "Total cell   : %15ld\n", total_cell);
    fprintf(writeFile, "Computed cell: %15ld\n", compute_cell);
    fprintf(writeFile, "Pruned cell  : %15ld\n", prune_cell);
    fclose(writeFile);
#endif
    for (int i = 0; i < R_max + 1; i++) {
        delete[] mem[i];
    }

    delete[] mem;
    delete trie;
    // delete node_stack;
    return join_result;
}

vector<vector<int>> taste_join(vector<u32string> &R, vector<u32string> &S, int delta) {
#ifdef JOIN_INFO
    int pruned_count = 0;
    int remain_count = 0;
    system_clock::time_point tps;
    system_clock::time_point tpe;
    duration<double> trie_build_time = duration<double>::zero();
    duration<double> filter_time = duration<double>::zero();
    duration<double> cal_dist_time = duration<double>::zero();
    duration<double> total_time = duration<double>::zero();
    long int total_line = 0;
    long int compute_line = 0;
    long int prune_line = 0;
    long int total_cell = 0;
    long int compute_cell = 0;
    long int prune_cell = 0;
#endif

    u32string r;
    u32string s;
    // int s_len;
    int r_len;

    vector<vector<int>> join_result;
    for (int i = 0; i < (int)R.size(); i++) {
        vector<int> res(delta + 1, 0);
        join_result.push_back(res);
    }

#ifdef JOIN_INFO
    tps = system_clock::now();
    cout << "trie build start" << endl;
#endif
    /****** create inverted dictionary for trie *************/
    unordered_map<u32string, vector<int> *> inv_dict;
    inv_dict.reserve(TASTE_HASH_SIZE);
    int n_part = delta + 1;
    int l_part;
    int remain;

    inv_dict[U"\0"] = new vector<int>();
    vector<u32string> parts_vec;
    unordered_set<u32string> parts;
    parts.reserve(TASTE_HASH_SIZE);

    int si;
    int ei;
    u32string part;

    for (int r_id = 0; r_id < (int)R.size(); r_id++) {
        r = R[r_id];
        r_len = r.size();
        if (r_len < n_part) {
            inv_dict[U"\0"]->push_back(r_id);
        } else {
            l_part = r_len / n_part;
            remain = r_len - l_part * n_part;
#ifdef JOIN_DEBUG
            if (r_len < n_part) {
                assert(false);
            }
#endif
            si = 0;
            ei = 0;
            for (int i = 0; i < n_part; i++) {
                si = ei;
                ei += l_part;
                if (i < remain) {
                    ei += 1;
                }
                part = r.substr(si, ei - si);
                if (inv_dict.find(part) == inv_dict.end()) {
                    inv_dict[part] = new vector<int>();
                }
                inv_dict[part]->push_back(r_id);
            }
#ifdef JOIN_DEBUG
#endif
            // parts_vec = wSED::partition_string(r, n_part);
            // parts = util::convertVec2Set(parts_vec);

            // for (auto part : parts) {
            //     if (inv_dict.find(part) == inv_dict.end()) {
            //         inv_dict[part] = new vector<int>();
            //     }
            //     inv_dict[part]->push_back(i);
            // }
        }
        // cout << "query: " << query << " [curr/total]: [" << i + 1 << "/" << queries.size() << "]" << endl;
    }

    /****** create trie structure *************/
    TriePart *trie = new TriePart();
    vector<int> *inv_list;
    for (auto iter = inv_dict.begin(); iter != inv_dict.end(); iter++) {
        part = iter->first;
        inv_list = iter->second;
        // if (partition == "roz") {
        //     cout << "roz trie inv debug:" << endl
        //          << "\t";
        //     util::printVec(*inv_list);
        // }
        trie->add_key_val(part, inv_list);
    }

#ifdef JOIN_INFO
    cout << "trie build done" << endl;
    tpe = system_clock::now();
    trie_build_time += (tpe - tps);
    cout << "start join" << endl;
#endif

    unordered_set<int> candidates;
    vector<int> cand_vec;

    int sed;
    for (int sid = 0; sid < (int)S.size(); sid++) {
        s = S[sid];
        // cout << "rec: " << rec << endl;
#ifdef JOIN_INFO
        // cout << "start get candidate" << endl;
        tps = system_clock::now();
#endif
        candidates = trie->candidate(s);
#ifdef JOIN_INFO
        // vector<int> cand_vec(candidates.begin(), candidates.end());
        // util::printVec(cand_vec);
        tpe = system_clock::now();
        filter_time += (tpe - tps);

        remain_count += int(candidates.size());
        pruned_count += int(R.size()) - int(candidates.size());

        tps = system_clock::now();
#endif
        for (int cand_idx : candidates) {
            r = R[cand_idx];
            sed = wSED::SED(r, s);
#ifdef JOIN_INFO
            compute_line += r.size();
            compute_cell += r.size() * s.size();
#endif

            // for (int k = sed; k < int(delta) + 1; k++) {
            //     join_result[cand_idx][k] += 1;
            // }
            if (sed <= delta) {
                join_result[cand_idx][sed] += 1;
            }
        }
#ifdef JOIN_DEBUG
        for (int r_id = 0; r_id < (int)R.size(); ++r_id) {
            r = R[r_id];
            if (candidates.find(r_id) == candidates.end()) {
                if ((int)r.size() <= delta) {
                    assert(false);
                }
                sed = wSED::SED(r, s);
                if (sed <= delta) {
                    assert(false);
                }
            }
        }
#endif

#ifdef JOIN_INFO
        tpe = system_clock::now();
        cal_dist_time += (tpe - tps);
#endif
    }
    for (int rid = 0; rid < (int)R.size(); rid++) {  // accumlation make slightly faster
        for (int i = 1; i < delta + 1; i++) {
            join_result[rid][i] += join_result[rid][i - 1];
        }
    }

#ifdef JOIN_INFO
    vector<int> r_lens;
    for (auto r : R) {
        r_lens.push_back(r.length());
    }
    vector<int> s_lens;
    for (auto s : S) {
        s_lens.push_back(s.length());
    }

    int R_max = *max_element(begin(r_lens), end(r_lens));
    // int S_max = *max_element(begin(s_lens), end(s_lens));

    long int r_total = accumulate(r_lens.begin(), r_lens.end(), 0);
    long int s_total = accumulate(s_lens.begin(), s_lens.end(), 0);

    total_line = r_total * S.size();
    prune_line = total_line - compute_line;

    total_cell = r_total * s_total;
    prune_cell = total_cell - compute_cell;
    printf("r_max        : %15d\n", R_max);
    printf("r_total      : %15ld\n", r_total);
    printf("s_total      : %15ld\n", s_total);
    printf("Total line   : %15ld\n", total_line);
    printf("Computed line: %15ld\n", compute_line);
    printf("Pruned line  : %15ld\n", prune_line);
    printf("Total cell   : %15ld\n", total_cell);
    printf("Computed cell: %15ld\n", compute_cell);
    printf("Pruned cell  : %15ld\n", prune_cell);

    string statFileName = "stat/" + args_str + ".txt";
    auto writeFile = fopen(statFileName.c_str(), "w");
    fprintf(writeFile, "[%s] %s\n",
            getTimeStamp().c_str(), args_str.c_str());
    fprintf(writeFile, "Total line   : %15ld\n", total_line);
    fprintf(writeFile, "Computed line: %15ld\n", compute_line);
    fprintf(writeFile, "Pruned line  : %15ld\n", prune_line);
    fprintf(writeFile, "Total cell   : %15ld\n", total_cell);
    fprintf(writeFile, "Computed cell: %15ld\n", compute_cell);
    fprintf(writeFile, "Pruned cell  : %15ld\n", prune_cell);
    fclose(writeFile);
#endif

    delete trie;
    return join_result;
}

tuple<vector<int>, vector<int>> gen_min_hash_coefficient(int L, int seed) {
    mt19937 gen(seed);
    uniform_int_distribution<int> dis_A(1, PRIME - 1);  // closed interval
    uniform_int_distribution<int> dis_B(0, PRIME - 1);
    vector<int> A;
    vector<int> B;

    for (int i = 0; i < L; ++i) {
        A.push_back(dis_A(gen));
        B.push_back(dis_B(gen));
    }
    return make_tuple(A, B);
}

vector<vector<int>> gen_min_hash_permutation_from_coefficients(int n, vector<int> &A, vector<int> &B) {
    assert(A.size() == B.size());

    int L = int(A.size());
    vector<vector<int>> order_lists;

    for (int i = 0; i < L; ++i) {
        int a = A[i];
        int b = B[i];
        vector<int> order_list(n);
        // vector<tuple<int, int>> hashes;
        priority_queue<TI2, vector<TI2>, greater<TI2>> queue;

        for (int j = 0; j < n; ++j) {
            // hashes.push_back(make_tuple(, (a * j + b) % PRIME));
            int hash = (a * j + b) % PRIME;
            queue.push(make_tuple(hash, j));
        }

        int order_index = 0;
        // make_heap(hashes.begin(), hashes.end());

        while (queue.size() > 0) {
            TI2 tie = queue.top();
            queue.pop();
            // int key = get<0>(tie);
            int val = get<1>(tie);
            order_list[val] = order_index;
            // int hash = pop_heap(hashes.begin(), hashes.end());
            // order_dict[hash] = order_index;
            order_index += 1;
        }
        order_lists.push_back(order_list);
    }
    return order_lists;
}

vector<vector<int>> gen_min_hash_permutation(int n, int L, int seed) {
    assert(n < PRIME);
    tuple<vector<int>, vector<int>> coefficients = gen_min_hash_coefficient(L, seed);
    vector<int> A = get<0>(coefficients);
    vector<int> B = get<1>(coefficients);

    vector<vector<int>> permutations = gen_min_hash_permutation_from_coefficients(n, A, B);

    return permutations;
}

vector<vector<int>> topk_join(vector<u32string> &R, vector<u32string> &S, int delta, int prune_level, int q = 3, int L = 50, int seed = 0) {
    // prune_level
    // 0: Topk-LB
    // 1: Topk-SPLIT
    // 2: Topk-INDEX
#ifdef JOIN_INFO
    // system_clock::time_point tps;
    // system_clock::time_point tpe;
    duration<double> duration_time = duration<double>::zero();
    long int total_line = 0;
    long int compute_line = 0;
    long int prune_line = 0;
    long int total_cell = 0;
    long int compute_cell = 0;
    long int prune_cell = 0;
#endif

    vector<vector<int>> join_result;
    for (int i = 0; i < (int)R.size(); i++) {
        vector<int> res(delta + 1, 0);
        join_result.push_back(res);
    }

    int lb;
    u32string query;
    u32string rec;
    int q_len;

    u32string q_qgram;
    u32string r_qgram;

    vector<tuple<vector<int> *, int>> matched_list_ptr;
    vector<int> *pi_list_ptr;
    vector<int> *pu_list_ptr;
    // int pi, pu;
    int rj, rv;
    int pdiff;
    int rdiff;
    int entry_value;
    int sed;

    vector<int> r_lens;
    for (auto r : R) {
        r_lens.push_back(r.length());
    }
    vector<int> s_lens;
    for (auto s : S) {
        s_lens.push_back(s.length());
    }

    int R_max = *max_element(begin(r_lens), end(r_lens));
    int S_max = *max_element(begin(s_lens), end(s_lens));

    /********* for DYN-lB *************/
    int **mem = new int *[R_max + 1];
    for (int i = 0; i < R_max + 1; i++) {
        mem[i] = new int[S_max + 1];
    }

    /**************** init Q ******************/
    unordered_map<u32string, vector<int>, CustomStringHash> q_qgram_dict;
    q_qgram_dict.reserve(TOPK_HASH_SIZE);

    /****************** for Best G' ****************/
    vector<int> *inv_list;
    int n = S.size();
    unordered_map<u32string, vector<int> *, CustomStringHash> r_inverted_list;  // L_(g_i) (with decreasing order)
    r_inverted_list.reserve(TOPK_HASH_SIZE);
    // unordered_map<u32string, vector<int> *, CustomStringHash> r_inverted_hash(10000);  // to estimate union of set
    unordered_map<u32string, int *, CustomStringHash> r_inverted_hash;  // to estimate union of set
    r_inverted_hash.reserve(TOPK_HASH_SIZE);
    vector<vector<int>> permutations;
    // vector<int> default_hash(L + 1, n + 1);  // size, val
    // default_hash[0] = 0;
    int *default_hash = nullptr;
    int hash;
    vector<int> *permutation;

    /********* for dynamic programing in Best G' *******/
    int rho = delta + 1;       // threshold for pruning a group of strings
    double **mem_u = nullptr;  // (1-index) u[i, t]: the minimum value of L(\theta(i,t))
    int ***mem_mh = nullptr;   // (1-index, 1-index, 0-index) min_hash of \theta(i,t)
    int **mem_pos = nullptr;   // (1-index) j* eq (7)

    int **d_hash_list = nullptr;  // (1-index) it contains a s_id list for each the i-th q-gram q[i:i+q]
    vector<int> S_ids;
    for (int i = 0; i < (int)S.size(); i++) {
        S_ids.push_back(i);
    }
    vector<int> S_ids_filtered;
    vector<int> *candidates;  // candidate ids in S
    // vector<tuple<int, u32string>> best_G;
    int max_pos;  // inclusive (1-index)
    int min_pos;
    int curr_pos;
    int last_pos;
    int t_idx;
    vector<int> Q_prime_pos;  // best Q' (1-index)
    double curr_count;
    int *curr_minHash;
    int curr_max_pos;
    int prev_min_pos;
    double prev_count;
    int *prev_minHash;
    double est_jaccard;
    double union_count;
    int common_count;
    int h1, h2;
    int *union_mh = nullptr;  // initailized later
    int *tmp_mh;
    vector<vector<int>> cand_lists;        // posting lists for best-G'
    unordered_set<u32string> best_qgrams;  // qgrams for best-G'
    best_qgrams.reserve(TOPK_HASH_SIZE);
    vector<int> *min_list;  // posting list in me
    TI2 min_element;
    vector<int> *sorted_list;  // posting list for q-qgram (decreasing order)
    priority_queue<TI2, vector<TI2>, greater<TI2>> heap;

    int r_id_prev;
    int cand_r_id;
    int cand_rid;
    int next_rid;
    int l_id;
    int pos_q;

    if (prune_level >= 1) {  // for Topk-SPLIT and Topk-INDEX
        /****************** inverted list *************/
        for (int s_id = (int)S.size() - 1; s_id >= 0; --s_id) {  // decreasing order for merge list
            rec = S[s_id];
            for (int s = 0; s < (int)rec.size() - q + 1; ++s) {
                r_qgram = rec.substr(s, q);
                if (r_inverted_list.find(r_qgram) == r_inverted_list.end()) {
                    r_inverted_list[r_qgram] = new vector<int>;
                    inv_list = r_inverted_list[r_qgram];
                    inv_list->push_back(s_id);
                } else {
                    inv_list = r_inverted_list[r_qgram];
                }

                if (inv_list->back() != s_id) {  // deduplication
                    inv_list->push_back(s_id);
                }
            }
        }
#ifdef JOIN_DEBUG
        // check inverted list
        // for (int s_id = 0; s_id < (int)S.size(); ++s_id) {  // decreasing order for merge list
        //     rec = S[s_id];
        //     for (int s = 0; s < (int)rec.size() - q + 1; ++s) {
        //         r_qgram = rec.substr(s, q);
        //         if (r_inverted_list.find(r_qgram) == r_inverted_list.end()) {
        //             assert(false);
        //         } else {
        //             inv_list = r_inverted_list[r_qgram];
        //         }
        //         if (std::find(inv_list->begin(), inv_list->end(), s_id) == inv_list->end()) {
        //             assert(false);
        //         }
        //     }
        // }
        // cout << "check inverted list done" << endl;
#endif

        /****************** min hash ****************/
        permutations = gen_min_hash_permutation(n, L, seed);
        for (auto item : r_inverted_list) {
            r_qgram = get<0>(item);
            inv_list = get<1>(item);
            // r_inverted_hash[r_qgram] = new vector<int>(L + 1, 0);  //size, val
            r_inverted_hash[r_qgram] = new int[L + 1];  // size, val
            // memset(r_inverted_hash[r_qgram], 0, sizeof(int) * (L + 1));
            // (*r_inverted_hash[r_qgram])[0] = (int)inv_list->size();
            r_inverted_hash[r_qgram][0] = (int)inv_list->size();
            for (int i = 1; i < L + 1; ++i) {
                permutation = &permutations[i - 1];
                hash = n;
                for (int item : *inv_list) {
                    hash = MIN(hash, (*permutation)[item]);
                }
                // (*r_inverted_hash[r_qgram])[i] = hash;
                r_inverted_hash[r_qgram][i] = hash;
            }
        }

        /********* for dynamic programing in Best G' *******/
        mem_u = new double *[R_max + 1];
        mem_mh = new int **[R_max + 1];
        mem_pos = new int *[R_max + 1];

        // initialize
        for (int i = 0; i < R_max + 1; i++) {  // (1-index)
            mem_u[i] = new double[rho + 1];    // (1-index)
            fill_n(mem_u[i], rho + 1, 0);
            mem_mh[i] = new int *[rho + 1];
            // for (int t = 0; t < rho + 1; t++) {  // (1-index)
            //     mem_mh[i][t] = new int[L];          // (0-index)
            // }
            for (int t = 2; t < rho + 1; t++) {  // (1-index) t=1 just pointing
                mem_mh[i][t] = new int[L];       // (0-index)
                fill_n(mem_mh[i][t], L, 0);
            }
            mem_pos[i] = new int[rho + 1];  // (1-index)
        }
        union_mh = new int[L];
        // for (int i = 0; i < L; ++i) {
        //     union_mh[i] = 0;
        // }
        default_hash = new int[L + 1];
        for (int i = 1; i < L + 1; ++i) {
            default_hash[i] = n + 1;
        }
        default_hash[0] = 0;
        d_hash_list = new int *[R_max + 1];  // (1-index) it contains a s_id list for each the i-th q-gram q[i:i+q]
    }

    for (int q_id = 0; q_id < (int)R.size(); ++q_id) {
#ifdef JOIN_DEBUG
        // // check inverted list
        // for (int s_id = 0; s_id < (int)S.size(); ++s_id) {  // decreasing order for merge list
        //     rec = S[s_id];
        //     for (int s = 0; s < (int)rec.size() - q + 1; ++s) {
        //         r_qgram = rec.substr(s, q);
        //         if (r_inverted_list.find(r_qgram) == r_inverted_list.end()) {
        //             assert(false);
        //         } else {
        //             inv_list = r_inverted_list[r_qgram];
        //         }
        //         if (std::find(inv_list->begin(), inv_list->end(), s_id) == inv_list->end()) {
        //             assert(false);
        //         }
        //     }
        // }
        // cout << "check inverted list done: (q_id) " << q_id << endl;
#endif
        query = R[q_id];
        q_len = (int)query.size();
        if ((prune_level >= 1) && (delta + 1 <= (int)query.size() / q)) {  // Best-G' for Topk-SPLIT and Topk-INDEX
            candidates = &S_ids_filtered;
            candidates->clear();
            // d_hash_list.clear();
            max_pos = q_len - q + 1;                       // inclusive (1-index)
            for (int pos = 1; pos < max_pos + 1; pos++) {  // 1-index
                q_qgram = query.substr(pos - 1, q);
                if (r_inverted_hash.find(q_qgram) != r_inverted_hash.end()) {
                    // d_hash_list.push_back(r_inverted_hash[q_qgram]);
                    d_hash_list[pos] = r_inverted_hash[q_qgram];
                } else {
                    d_hash_list[pos] = default_hash;
                    // d_hash_list.push_back(&default_hash);
                }
            }
            // best_G.clear();

            // initialize for t = 1 (only one q-gram)
            for (int pos = 1; pos < max_pos + 1; pos++) {  // 1-index
                // curr_count = (double)(*d_hash_list[pos - 1])[0];
                curr_count = (double)d_hash_list[pos][0];  // d_hash_list : 1-index
                mem_u[pos][1] = curr_count;                // 1-index
                // for (int j = 0; j < L; j++) {
                //     mem_mh[pos][1][j] = d_hash_list[pos - 1][j + 1];
                // }
                mem_mh[pos][1] = &(d_hash_list[pos][1]);
                // no need mem_pos[pos][1]
            }

            for (int t = 2; t < rho + 1; t++) {
                min_pos = (t - 1) * q + 1;       // (1-index)
                prev_min_pos = (t - 2) * q + 1;  // (1-index)

                for (int pos = min_pos; pos < max_pos + 1; pos++) {
                    curr_count = (double)d_hash_list[pos][0];  // d_hash_list : 1-index
                    // curr_minHash = &(*d_hash_list[i - 1])[1];
                    curr_minHash = &(d_hash_list[pos][1]);

                    curr_max_pos = pos - q;                                  // inclusive (1-index)
                    mem_u[pos][t] = n + 1;                                   // 1-index
                    for (int k = prev_min_pos; k < curr_max_pos + 1; k++) {  // 1-index
                        prev_count = mem_u[k][t - 1];
                        prev_minHash = mem_mh[k][t - 1];

                        common_count = 0;
                        for (int hash_idx = 0; hash_idx < L; hash_idx++) {
                            h1 = curr_minHash[hash_idx];
                            h2 = prev_minHash[hash_idx];

                            if (h1 == h2) {
                                common_count += 1;
                            }
                            union_mh[hash_idx] = MIN(h1, h2);
                        }

                        est_jaccard = (double)common_count / L;
                        union_count = (double)(curr_count + prev_count) / (est_jaccard + 1);

                        if (mem_u[pos][t] > union_count) {
                            mem_u[pos][t] = union_count;
                            tmp_mh = mem_mh[pos][t];
                            mem_mh[pos][t] = union_mh;
                            union_mh = tmp_mh;
                        }
                        mem_pos[pos][t] = k;
                    }
                }
            }

            /***** Find G' by backtracking ***********/
            min_pos = (rho - 1) * q + 1;  // 1-index
            curr_count = n + 1;           // assign infinite value
            // assert(min_pos <= max_pos);

            // find last position
            last_pos = min_pos;
            for (int i = min_pos; i < max_pos + 1; i++) {
                if (curr_count > mem_u[i][rho]) {
                    curr_count = mem_u[i][rho];
                    last_pos = i;
                }
            }
            // if (q_id < 10) {
            //     printf("est: %f\n", curr_count);
            //     util::print2dArr(mem_u, q_len, rho + 1);
            // }

            Q_prime_pos.clear();  // Q_prime_pos (decreasing order) (1-index)
            curr_pos = last_pos;
            Q_prime_pos.push_back(curr_pos);  // for (size)th q-gram
            t_idx = rho;
            while (t_idx > 1) {                       // from (size-1)th to 1st q-gram
                curr_pos = mem_pos[curr_pos][t_idx];  // (t-1)th q-gram
                Q_prime_pos.push_back(curr_pos);
                t_idx -= 1;
            }

            S_ids_filtered.clear();
            if (prune_level >= 2) {
                // index part of TopK-INDEX
                cand_lists.clear();
#ifdef JOIN_DEBUG
                cout << "query: " << utf8::utf32to8(query) << endl;
#endif
                for (int idx_q = (int)Q_prime_pos.size() - 1; idx_q >= 0; --idx_q) {
                    pos_q = Q_prime_pos[idx_q];  // 1-index
                    q_qgram = query.substr(pos_q - 1, q);
#ifdef JOIN_DEBUG
                    cout << "q_qgram: " << utf8::utf32to8(q_qgram) << " pos_q: " << pos_q << endl;
#endif
                    if (r_inverted_list.find(q_qgram) != r_inverted_list.end()) {
                        cand_lists.push_back(*r_inverted_list[q_qgram]);
                    }
                }

#ifdef JOIN_DEBUG
                if (heap.size() > 0) {
                    assert(false);
                }
#endif
                // heap.clear();
                for (int l_id = 0; l_id < (int)cand_lists.size(); l_id++) {
                    sorted_list = &cand_lists[l_id];
                    // reverse(sorted_list.begin(), sorted_list.end());  // incresing order to decreasing order
                    heap.push(make_tuple(sorted_list->back(), l_id));  // decreasing order
                    sorted_list->pop_back();
                }
                r_id_prev = -1;
                while (heap.size() > 0) {
                    min_element = heap.top();
                    heap.pop();
                    cand_r_id = get<0>(min_element);
                    l_id = get<1>(min_element);
                    if (cand_r_id > r_id_prev) {
                        S_ids_filtered.push_back(cand_r_id);
                        r_id_prev = cand_r_id;
                    }
                    min_list = &cand_lists[l_id];

                    if (heap.size() == 0) {
                        while (min_list->size() > 0) {
                            cand_rid = min_list->back();
                            min_list->pop_back();
                            S_ids_filtered.push_back(cand_rid);
                        }
                    } else {
                        min_element = heap.top();
                        next_rid = get<0>(min_element);
                        while (min_list->size() > 0) {
                            cand_rid = min_list->back();
                            min_list->pop_back();

                            if (cand_rid < next_rid) {
                                S_ids_filtered.push_back(cand_rid);
                            } else if (cand_rid == next_rid) {
                                continue;
                            } else {
                                heap.push(make_tuple(cand_rid, l_id));
                                break;
                            }
                        }
                    }
                }
#ifdef JOIN_DEBUG
                cand_lists.clear();
                for (int idx_q = (int)Q_prime_pos.size() - 1; idx_q >= 0; --idx_q) {
                    pos_q = Q_prime_pos[idx_q];  // 1-index
                    q_qgram = query.substr(pos_q - 1, q);
                    if (r_inverted_list.find(q_qgram) != r_inverted_list.end()) {
                        cand_lists.push_back(*r_inverted_list[q_qgram]);
                    }
                }
                unordered_set<int> candidates_gt;
                for (int l_id = 0; l_id < (int)cand_lists.size(); l_id++) {
                    sorted_list = &cand_lists[l_id];
                    // reverse(sorted_list.begin(), sorted_list.end());  // incresing order to decreasing order
                    for (int r_id : *sorted_list) {
                        candidates_gt.insert(r_id);
                    }
                }
                vector<int> candidates_gt_list(candidates_gt.begin(), candidates_gt.end());
                sort(candidates_gt_list.begin(), candidates_gt_list.end());
                if (candidates->size() != candidates_gt_list.size()) {
                    assert(false);
                }
                for (int i = 0; i < (int)candidates->size(); ++i) {
                    int val1 = (*candidates)[i];
                    int val2 = candidates_gt_list[i];
                    if (val1 != val2) {
                        assert(false);
                    }
                }
#endif
            } else {
                // enumeration part of TopK-SPLIT
                best_qgrams.clear();
                for (int pos_q = (int)Q_prime_pos.size() - 1; pos_q >= 0; --pos_q) {
                    q_qgram = query.substr(pos_q, q);
                    best_qgrams.insert(q_qgram);
                }
                for (auto sid : S_ids) {
                    rec = S[sid];
                    for (auto q_qgram : best_qgrams) {
                        if (rec.find(q_qgram) != std::string::npos) {
                            S_ids_filtered.push_back(sid);
                            break;
                        }
                    }
                }
            }
            candidates = &S_ids_filtered;
        } else {
            candidates = &S_ids;
        }
        q_qgram_dict.clear();
        for (int p = 0; p < (int)query.size() - q + 1; p++) {
            q_qgram = query.substr(p, q);
            q_qgram_dict[q_qgram].push_back(p + 1);  // 1-index
        }
        q_len = query.size();
        // if (q_id < 10) {
        //     printf("%d: %d candidates\n", q_id + 1, candidates->size());
        // }
#ifdef JOIN_DEBUG
        for (int s_id = 0; s_id < (int)S.size(); ++s_id) {
            if (std::find(candidates->begin(), candidates->end(), s_id) == candidates->end()) {
                rec = S[s_id];
                sed = wSED::SED(query, rec);
                if (sed <= delta) {
                    assert(false);
                }
            }
        }
#endif

        for (auto s_id : *candidates) {
            rec = S[s_id];

            // lb = get_DYN_LB(q_qgram_dict, q_len, rec, q);

            /*************** DYN-LB *****************/
            matched_list_ptr.clear();
            lb = std::ceil((float)(q_len - q + 1) / (float)q);
            for (rj = 1; rj < (int)rec.size() - q + 2; ++rj) {
                r_qgram = rec.substr(rj - 1, q);
                if (q_qgram_dict.find(r_qgram) != q_qgram_dict.end()) {
                    pi_list_ptr = &q_qgram_dict[r_qgram];
                    for (auto pi : *pi_list_ptr) {
                        entry_value = std::ceil((double)(pi - 1) / (double)q);
                        for (auto matched : matched_list_ptr) {
                            pu_list_ptr = get<0>(matched);
                            rv = get<1>(matched);

                            rdiff = rj - rv;
                            for (auto pu : *pu_list_ptr) {
                                if (pu >= pi) {
                                    continue;
                                }
                                pdiff = pi - pu;
                                entry_value = MIN(entry_value, mem[pu][rv] + MAX(std::ceil((double)(pdiff - 1) / (double)q), abs(pdiff - rdiff)));
                            }
                        }
                        mem[pi][rj] = entry_value;
                        lb = MIN(lb, entry_value + std::ceil((float)(q_len - pi - q + 1) / (float)q));
                    }
                    matched_list_ptr.push_back(make_tuple(pi_list_ptr, rj));
                }
            }
#ifdef JOIN_DEBUG
            // sed = wSED::SED(query, rec);
            // if (lb > sed) {
            //     assert(false);
            // }
#endif
            if (lb > delta) {
                continue;
            }

            sed = wSED::SED(query, rec);
#ifdef JOIN_INFO
            compute_line += query.size();
            compute_cell += query.size() * rec.size();
#endif

            if (sed <= delta) {
                join_result[q_id][sed] += 1;
            }
        }
    }

    for (int rid = 0; rid < (int)R.size(); rid++) {  // accumlation make slightly faster
        for (int i = 1; i < delta + 1; i++) {
            join_result[rid][i] += join_result[rid][i - 1];
        }
    }
#ifdef JOIN_INFO
    long int r_total = accumulate(r_lens.begin(), r_lens.end(), 0);
    long int s_total = accumulate(s_lens.begin(), s_lens.end(), 0);

    total_line = r_total * S.size();
    prune_line = total_line - compute_line;

    total_cell = r_total * s_total;
    prune_cell = total_cell - compute_cell;
    printf("r_max        : %15d\n", R_max);
    printf("r_total      : %15ld\n", r_total);
    printf("s_total      : %15ld\n", s_total);
    printf("Total line   : %15ld\n", total_line);
    printf("Computed line: %15ld\n", compute_line);
    printf("Pruned line  : %15ld\n", prune_line);
    printf("Total cell   : %15ld\n", total_cell);
    printf("Computed cell: %15ld\n", compute_cell);
    printf("Pruned cell  : %15ld\n", prune_cell);

    string statFileName = "stat/" + args_str + ".txt";
    auto writeFile = fopen(statFileName.c_str(), "w");
    fprintf(writeFile, "[%s] %s\n",
            getTimeStamp().c_str(), args_str.c_str());
    fprintf(writeFile, "Total line   : %15ld\n", total_line);
    fprintf(writeFile, "Computed line: %15ld\n", compute_line);
    fprintf(writeFile, "Pruned line  : %15ld\n", prune_line);
    fprintf(writeFile, "Total cell   : %15ld\n", total_cell);
    fprintf(writeFile, "Computed cell: %15ld\n", compute_cell);
    fprintf(writeFile, "Pruned cell  : %15ld\n", prune_cell);
    fclose(writeFile);
#endif

    for (int i = 0; i < R_max + 1; i++) {
        delete[] mem[i];
    }
    delete[] mem;

    if (prune_level >= 1) {
        for (auto iter = r_inverted_list.begin(); iter != r_inverted_list.end(); ++iter) {
            delete iter->second;
        }
        for (auto iter = r_inverted_hash.begin(); iter != r_inverted_hash.end(); ++iter) {
            delete[] iter->second;
        }
        for (int i = 0; i < R_max + 1; ++i) {
            delete[] mem_u[i];
            // for (int j = 0; j < rho + 1; j++) {
            //     delete[] mem_mh[i][j];
            // }
            for (int j = 2; j < rho + 1; j++) {
                delete[] mem_mh[i][j];
            }
            delete[] mem_mh[i];
            delete[] mem_pos[i];
        }
        delete[] mem_u;
        delete[] mem_mh;
        delete[] mem_pos;
        delete[] union_mh;
        delete[] default_hash;
        delete[] d_hash_list;
    }

    return join_result;
}

vector<vector<int>> get_count_array(string algName, vector<u32string> &S_Q, vector<u32string> &S_D, int delta_M, bool prefix_mode, char32_t **S_Q_ptr = nullptr) {
    vector<vector<int>> join_result;
    vector<u32string> S_QP = S_Q;
    if (prefix_mode) {
        S_QP = distinct_prefix(S_Q);
    }
    // auto tps = system_clock::now();
    switch (str2inthash(algName.c_str())) {
        case str2inthash("NaiveGen"): {
            join_result = all_pair_join(S_QP, S_D, delta_M);
            break;
        }
        case str2inthash("SODDY"): {
            join_result = SODDY_join(S_Q.size(), S_D, delta_M, 2, prefix_mode, S_Q_ptr);
            break;
        }
        case str2inthash("TEDDY-S"): {
            join_result = TEDDY_join(S_Q, S_D, delta_M, 0, prefix_mode);
            break;
        }
        case str2inthash("TEDDY"): {
            join_result = TEDDY_join(S_Q, S_D, delta_M, 2, prefix_mode);
            break;
        }
        case str2inthash("TEDDY-R"): {
            join_result = Ablation_join(S_QP, S_D, delta_M);
            break;
        }
        case str2inthash("TASTE"): {
            join_result = taste_join(S_QP, S_D, delta_M);
            break;
        }
        case str2inthash("Qgram"): {
            int q = 2;
            join_result = topk_join(S_QP, S_D, delta_M, 2, q);
            break;
        }
        default: {
            cout << "invalid alg name is given: " << algName << endl;
            cout << "check input algorithm list" << endl;
            exit(0);
        }
    }
    // auto tpe = system_clock::now();
    // auto duration_time = duration<double>(tpe - tps).count();
    // cout << "join time: " << duration_time << endl;
    return join_result;
}

CT get_count_table(string algName, vector<u32string> S_Q, vector<u32string> S_D, int delta_M, bool prefix_mode, char32_t **S_Q_ptr = nullptr) {
    // auto tps = system_clock::now();
    vector<vector<int>> join_result = get_count_array(algName, S_Q, S_D, delta_M, prefix_mode, S_Q_ptr);
    // auto tpe = system_clock::now();
    // auto duration_time = duration<double>(tpe - tps).count();
    // cout << "count(arr) time: " << duration_time << endl;
    CT ct;
    vector<u32string> S_QP = S_Q;
    if (prefix_mode) {
        S_QP = distinct_prefix(S_Q);
    }
    Qry qry;
    for (int i = 0; i < int(S_QP.size()); ++i) {
        auto q = S_QP[i];
        for (int delta = 0; delta <= delta_M; ++delta) {
            qry = Qry(q, delta);
            ct[qry] = join_result[i][delta];
        }
    }
    return ct;
}

void write_train_instance(vector<u32string> &S_Q, int delta_M, CT ct, bool prefix_aug, string ofname) {
    u32stringstream ss_intv;
    ofstream writeFile;
    writeFile.open(ofname.c_str(), fstream::out);
    if (!writeFile) {
        cout << "is not open at " << ofname << endl;
        exit(0);
    }
    cout << "The training data is written as " << ofname << endl;

    // write header
    writeFile << "word, delta, card(s)" << endl;

    Qry qry;
    string q_csv;
    for (auto q : S_Q) {
        for (int delta = 0; delta <= delta_M; ++delta) {
            q_csv = u32string2string(csv_token(q));
            writeFile << q_csv << ',' << delta << ',';
            if (prefix_aug) {
                for (int i = 0; i < (int)q.size(); ++i) {
                    qry = Qry(q.substr(0, i + 1), delta);
                    if (i) {
                        writeFile << ":";
                    }
                    writeFile << ct[qry];
                }

            } else {
                qry = Qry(q, delta);
                writeFile << ct[qry];
            }
            writeFile << endl;
        }
    }

    writeFile.close();
}

void write_train_data(string algName, vector<u32string> &S_Q, vector<u32string> &S_D, int delta_M, bool prefix_aug, string ofname, char32_t **S_Q_ptr = nullptr) {
    // auto tps = system_clock::now();
    CT ct = get_count_table(algName, S_Q, S_D, delta_M, prefix_aug, S_Q_ptr);
    // auto tpe = system_clock::now();
    // auto duration_time = duration<double>(tpe - tps).count();
    // cout << "count time: " << duration_time << endl;

    // tps = system_clock::now();
#ifndef JOIN_INFO
    write_train_instance(S_Q, delta_M, ct, prefix_aug, ofname);
#endif
    // tpe = system_clock::now();
    // duration_time = duration<double>(tpe - tps).count();
    // cout << "write time: " << duration_time << endl;
}

}  // namespace join

#endif
