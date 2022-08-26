#include <algorithm>
#include <cassert>
#include <chrono>
#include <codecvt>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <iterator>
#include <locale>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "join.h"
#include "trie.h"
#include "utf8.h"
#include "util.h"
#include "wSED.h"
// #define __DEBUG__

using namespace std;
using namespace chrono;
using namespace wSED;
using namespace util;
using namespace join;

system_clock::time_point tps;
system_clock::time_point tpe;

void print_envs() {
#ifdef JOIN_INFO
    cout << "JOIN_INFO is defined" << endl;
#else
    cout << "JOIN_INFO is not defined" << endl;
#endif
    // #ifdef JOIN_DEBUG
    //     cout << "JOIN_DEBUG is defined" << endl;
    // #else
    //     cout << "JOIN_DEBUG is not defined" << endl;
    // #endif
}

string get_args_str(int argc, char *argv[]) {
    stringstream ss;
    for (int i = 1; i < argc; i++) {
        if (i > 1) {
            ss << " ";
        }
        ss << argv[i];
    }
    string args_str = ss.str();
    std::replace(args_str.begin(), args_str.end(), '/', '\\');
    return args_str;
}

pair<int, float> num_str_to_int_with_ratio(const char *num_str) {
    int n;
    float p = -1.0;
    if (strchr(num_str, '.') != NULL) {
        p = atof(num_str);
        n = 0;
    } else {
        n = atoi(num_str);
    }

    if (n == 0) {
        n = INT32_MAX;
    }

    return pair<int, float>(n, p);
}

int num_str_to_int(const char *num_str, const char *filePath) {
    int n;
    float p = -1.0;
    if (strchr(num_str, '.') != NULL) {
        p = atof(num_str);
        n = 0;
    } else {
        n = atoi(num_str);
    }

    if (n == 0) {
        n = INT32_MAX;
    }
    ifstream readFile(filePath, ios::in);
    if (!readFile.is_open()) {
        cout << "Could not open " << filePath << endl;
        exit(0);
    }

    // cout << "Read dataset" << endl;
    string line;
    int line_count = 0;
    while (!readFile.eof() && line_count < n) {
        getline(readFile, line);
        if (line.length() > 0) {
            line_count += 1;
        }
    }
    n = line_count;
    if (p > 0) {
        n = MAX(n * p, 1);
    }
    return n;
}

vector<u32string> read_string_data(const char *filePath, int n, bool is_sort) {
    // cout << "Read dataset" << endl;
    ifstream readFile(filePath, ios::in);
    if (!readFile.is_open()) {
        cout << "Could not open " << filePath << endl;
        exit(0);
    }
    string line;
    u32string u32line;
    int line_count = 0;
    vector<u32string> S;
    while (!readFile.eof()) {
        getline(readFile, line);
        auto end_it = utf8::find_invalid(line.begin(), line.end());
        if (end_it != line.end()) {
            cout << "Invalid UTF-8 encoding detected at line " << line_count << "\n";
            cout << "This part is fine: " << string(line.begin(), end_it) << "\n";
        }

        u32line = utf8::utf8to32(line);
        if (u32line.length() > 0) {
            S.push_back(u32line);
            line_count += 1;
        }
    }

    std::srand(0);
    std::random_shuffle(S.begin(), S.end());
    S.resize(n);
    if (is_sort) {
        sort(S.begin(), S.end());
    }
    //  for (int i = 0; i < MIN(int(S.size()), 10); ++i) {
    //      u32line = S[i];
    //      cout << u32line.size() << ", " << wSED::conv_string2code.to_bytes(u32line) << endl;
    //  }
    return S;
}

void write_log(string logFileName, string algName, int nq, int nr, int S_Q_size, int delta_M, bool prefix_mode, float time = -1.0, char const *mode = "w") {
    FILE *writeFile;
    // cout << logFileName << endl;

    writeFile = fopen(logFileName.c_str(), mode);
    if (time > 0.0) {
        printf("%6s %5s %11.3f\n",
               algName.c_str(), prefix_mode ? "prefix-aug" : "base", time);
    }
    fprintf(writeFile, "[%s] %6s %7d %7d %8d %3d %5s %11.3f\n",
            getTimeStamp().c_str(), algName.c_str(), nq, nr, S_Q_size, delta_M, prefix_mode ? "True" : "False", time);
    fclose(writeFile);
}

int main(int argc, char *argv[]) {
    // command line arguments
    // cout << "You have entered " << argc << " argument(s)" << endl;
    // for (int i = 0; i < argc; i++) {
    //     cout << argv[i] << " ";
    // }
    // cout << endl;
    if (argc < 7) {
        cout << "Please enter ./main [alg] [data path] [query path] [delta] [prefix] [trial_id]" << endl;
        cout << "example: ./main TEDDY data/DBLP.txt data/qs_DBLP.txt 3 0 0" << endl;
        return 0;
    }

    int fail;
    fail = std::system("mkdir -p res");
    fail = std::system("mkdir -p stat");
    fail = std::system("mkdir -p time");
    if (fail) {
        cout << "fail to create dir" << endl;
        return 0;
    }

    // print_envs();

    string args_str = get_args_str(argc, argv);
    join::args_str = args_str;
    // cout << args_str << endl;

    // logging arguments
    ofstream wfstat;
    // wfstat.open("log.txt", fstream::out | fstream::app);
    // wfstat << "[" << util::getTimeStamp().c_str() << "] ";
    // wfstat << args_str << endl;

    // read arguments
    const char *arg_nr = "1.0";
    const char *arg_nq = "1.0";
    const char *arg_alg = argv[1];
    const char *arg_rec = argv[2];
    const char *arg_qry = argv[3];
    const char *arg_del = argv[4];
    const char *arg_prfx = argv[5];
    // const char *arg_trial = argv[6];

    int nr = num_str_to_int(arg_nr, arg_rec);
    vector<u32string> S_D = read_string_data(arg_rec, nr, true);
    int nq = num_str_to_int(arg_nq, arg_qry);
    vector<u32string> S_Q = read_string_data(arg_qry, nq, false);
    vector<u32string> S_P = distinct_prefix(S_Q);
    int np = (int)S_P.size();
    _n_prfx_ = np;
    int nc = 0;
    for (auto q : S_Q) {
        nc += q.size();
    }
    // cout << nq << ", " << nr << ", " << np << ", " << nc << endl;

    string algName = arg_alg;

    string queryFileName = remove_ext(basename(string(arg_qry)));

    int delta_M = stoi(arg_del);
    bool prefix_mode = stoi(arg_prfx);
    // if (prefix_mode) {
    //     cout << "prefix_mode" << endl;
    // }

    tps = system_clock::now();
    sort(S_Q.begin(), S_Q.end());
    tpe = system_clock::now();
    char32_t **S_Q_ptr = new char32_t *[nq];
    char32_t *S_Q_pack = new char32_t[nc + nq];
    int ch_i = 0;
    for (int i = 0; i < (int)S_Q.size(); ++i) {
        auto q = S_Q[i];
        S_Q_ptr[i] = &S_Q_pack[ch_i];
        S_Q_pack[ch_i] = q.size();
        ch_i++;
        for (int j = 0; j < (int)q.size(); ++j) {
            S_Q_pack[ch_i] = q[j];
            ch_i++;
        }
    }
    double sort_time = duration<double>(tpe - tps).count();
    // cout << "sort time: " << sort_time << endl;
    // cout << "num S_Q, S_D [" << S_Q.size() << ", " << S_D.size() << "]" << endl;

    string logFileName = "time/" + args_str + ".txt";
    write_log(logFileName, algName, nq, nr, int(S_Q.size()), delta_M, prefix_mode, -1.0);

    // cout << "Start algorihtm" << endl;
    double duration_time = 0.0;
    vector<vector<int>> join_result;

    string save_path = get_save_name(queryFileName, algName, nq, nr, delta_M, prefix_mode, np);
    tps = system_clock::now();
    write_train_data(algName, S_Q, S_D, delta_M, prefix_mode, save_path, S_Q_ptr);
    tpe = system_clock::now();

    // print a small part of result and execution time
    duration_time = duration<double>(tpe - tps).count();
    if (algName.find("soddy") != string::npos) {
        duration_time += sort_time;
    }
    // cout << algName << " sort time: " << sort_time << "s" << endl;
    // cout << algName << " join time: " << duration_time << "s" << endl;

    write_log(logFileName, algName, nq, nr, int(S_Q.size()), delta_M, prefix_mode, duration_time);
    delete[] S_Q_ptr;
    delete[] S_Q_pack;
    return 0;
}
