#ifndef util_FOR_CPP_2675DCD0_9480_4c0c_B92A_CC14C027B731
#define util_FOR_CPP_2675DCD0_9480_4c0c_B92A_CC14C027B731

#include <chrono>
#include <iomanip>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "utf8.h"

using namespace std;
using namespace chrono;
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

// typedef std::basic_istringstream<char32_t, std::char_traits<char32_t>, std::allocator<char32_t>> u32istringstream;
// typedef std::basic_string<char32_t, std::char_traits<char32_t>, std::allocator<char32_t>> _u32string;
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

typedef std::basic_istream<char32_t> u32istream;
typedef std::basic_istringstream<char32_t> u32istringstream;
typedef std::basic_stringstream<char32_t> u32stringstream;
typedef std::pair<u32string, int> Qry;
typedef std::unordered_map<Qry, int, pair_hash> CT;

namespace util {

constexpr unsigned int str2inthash(const char* str, int h = 0) {
    return !str[h] ? 5381 : (str2inthash(str, h + 1) * 33) ^ str[h];
}

string getTimeStamp() {
    std::time_t currTime = system_clock::to_time_t(system_clock::now());
    string currTimeStamp = strtok(std::ctime(&currTime), "\n");
    return currTimeStamp;
}

void del_last_line(const char* file_name) {
    cout << "delete last line of " << file_name << endl;
    ifstream fin(file_name);
    ofstream fout;
    fout.open("temp.txt", ios::out);

    char ch;
    int line = 1;
    while (fin.get(ch)) {
        if (ch == '\n')
            line++;
    }
    fin.close();
    fin.open(file_name);

    int n = line - (ch == '\n');
    line = 1;
    while (fin.get(ch)) {
        if (ch == '\n')
            line++;
        if (line != n)
            fout << ch;
    }
    fout.close();
    fin.close();
    remove(file_name);
    rename("temp.txt", file_name);
}

vector<u32string> split_string(u32string& x, char32_t delim = ' ') {
    vector<u32string> output;
    u32istringstream iss(x);
    u32string s;
    while (getline(iss, s, delim)) {
        output.push_back(s);
    }
    return output;
}
vector<string> split_string(string& x, char delim = ' ') {
    vector<string> output;
    istringstream iss(x);
    string s;
    while (getline(iss, s, delim)) {
        output.push_back(s);
    }
    return output;
}
std::vector<std::string> splitpath(
    const std::string& str, const std::set<char> delimiters) {
    std::vector<std::string> result;

    char const* pch = str.c_str();
    char const* start = pch;
    for (; *pch; ++pch) {
        if (delimiters.find(*pch) != delimiters.end()) {
            if (start != pch) {
                std::string str(start, pch);
                result.push_back(str);
            } else {
                result.push_back("");
            }
            start = pch + 1;
        }
    }
    result.push_back(start);

    return result;
}

string basename(string filePath) {
    set<char> delims{'\\', '/'};  // windows and linux
    string basename = splitpath(filePath.c_str(), delims).back();
    return basename;
}

string remove_ext(string filePath) {
    int lastindex = filePath.find_last_of(".");
    string filename = filePath.substr(0, lastindex);
    return filename;
}

string get_ext(string filePath) {
    int lastindex = filePath.find_last_of(".");
    string ext = filePath.substr(lastindex + 1);
    return ext;
}

template <typename _CharT>
std::string vector2string(std::vector<_CharT>& vec) {
    stringstream ss;
    ss << "[";
    bool first = true;
    for (_CharT x : vec) {
        if (!first) {
            ss << ", ";
        } else {
            first = false;
        }
        ss << x;
    }
    ss << "]";
    return ss.str();
}

u32string csv_token(u32string& x) {
    // stringstream ss_csv;
    u32stringstream ss_csv;
    if (x.find(U',') != string::npos || x.find(U'\"') != string::npos) {
        ss_csv << U'\"';
        for (char32_t c : x) {
            if (c == U'\"') {
                ss_csv << U"\"\"";
            } else {
                ss_csv << c;
            }
        }
        ss_csv << U'\"';
        // cout << ss_csv.str() <<endl;
        x = ss_csv.str();
    }
    // cout << x << endl;
    return x;
}

string csv_token(string& x) {
    stringstream ss_csv;
    if (x.find(',') != string::npos || x.find('\"') != string::npos) {
        ss_csv << '\"';
        for (char c : x) {
            if (c == '\"') {
                ss_csv << "\"\"";
            } else {
                ss_csv << c;
            }
        }
        ss_csv << '\"';
        // cout << ss_csv.str() <<endl;
        x = ss_csv.str();
    }
    // cout << x << endl;
    return x;
}

string get_save_name(string filename, string name, int nq, int nr, int delta_M, bool prefix_mode, int np) {
    stringstream filename_stream;

    // filename_stream << "res/" << filename << '_' << name << '_' << setfill('0') << setw(7) << nq << '_' << setfill('0') << setw(7) << nr;
    filename_stream << "res/" << filename << '_' << name;
    filename_stream << '_' << setfill('0') << setw(2) << delta_M;
    if (prefix_mode) {
        // filename_stream << '_' << setfill('0') << setw(8) << np;
        filename_stream << "_prefix";
    } else {
        filename_stream << "_base";
    }
    filename_stream << ".txt";
    string save_path = filename_stream.str();

    return save_path;
}

void save_exp_result(string filename, string name, int nq, int nr, int delta, vector<u32string> R, vector<vector<int>> join_result, bool prefix_mode = false) {
    stringstream filename_stream;
    // stringstream ss_intv;
    u32stringstream ss_intv;

    filename_stream << "res/" << filename << '_' << name << '_' << setfill('0') << setw(7) << nq << '_' << setfill('0') << setw(7) << nr;
    filename_stream << '_' << setfill('0') << setw(2) << delta;
    if (prefix_mode) {
        filename_stream << '_' << setfill('0') << setw(8) << join_result.size();
        filename_stream << "_prfx";
    }
    filename_stream << ".txt";
    string save_path = filename_stream.str();
    ofstream writeFile;
    writeFile.open(save_path.c_str(), fstream::out);
    if (!writeFile) {
        cout << "is not open at " << save_path << endl;
        exit(0);
    }
    cout << "saving at " << save_path << endl;
    cout << "R size:" << R.size() << endl;
    cout << "join result size:" << join_result.size() << endl;
    if (not prefix_mode) {
        assert(R.size() == join_result.size());
    }

    // write header
    ss_intv << U"word";
    for (int i = 0; i < delta + 1; i++) {
        ss_intv << U',' << utf8::utf8to32(to_string(i));
    }
    ss_intv << U'\n';
    writeFile << utf8::utf32to8(ss_intv.str());

    // write results
    for (int i = 0; i < (int)R.size(); i++) {
        ss_intv.str(U"");

        ss_intv << csv_token(R[i]);
        for (int val : join_result[i]) {
            ss_intv << U',' << utf8::utf8to32(to_string(val));
        }

        ss_intv << U'\n';
        if (i < MIN((int)R.size(), 10)) {
            cout << utf8::utf32to8(ss_intv.str());
        }

        string line = utf8::utf32to8(ss_intv.str());
        writeFile << line;
    }
    writeFile.close();
}
void save_exp_result(string filename, string name, int n, int delta, vector<string> R, vector<vector<int>> trie_intval_result) {
    stringstream filename_stream;
    stringstream ss_intv;
    filename_stream << "res/" << filename << '_' << name << '_' << n << '_' << delta << ".txt";
    string save_path = filename_stream.str();
    FILE* writeFile = fopen(save_path.c_str(), "w");
    cout << "saving at " << save_path << endl;
    // ofstream writeStream(filename_stream.str().c_str(), ios::out);
    cout << "R size:" << R.size() << endl;
    cout << "trie_intval_result size:" << trie_intval_result.size() << endl;
    ss_intv << "word";
    for (int i = 0; i < delta + 1; i++) {
        ss_intv << ',' << i;
    }
    ss_intv << endl;
    fprintf(writeFile, ss_intv.str().c_str(), "w");
    for (int i = 0; i < (int)R.size(); i++) {
        ss_intv.str("");

        // csv save;

        ss_intv << csv_token(R[i]);
        for (int val : trie_intval_result[i]) {
            ss_intv << ',' << val;
        }

        ss_intv << endl;
        if (i < MIN((int)R.size(), 10)) {
            cout << ss_intv.str();
        }

        // cout<< i << ' ' << ss.str() << endl;
        // cout << i << ' ' << ss.str() << endl;
        // writeStream << i << ' ' << ss.str() << endl;
        fprintf(writeFile, ss_intv.str().c_str(), "w");
    }
    fclose(writeFile);
}

template <typename _CharT>
unordered_set<_CharT> convertVec2Set(vector<_CharT> vec) {
    unordered_set<_CharT> output(vec.begin(), vec.end());
    return output;
}

template <typename _CharT>
void printVec(const vector<_CharT>& vec, bool is_newline = true) {
    cout << '[';
    for (auto itr = vec.begin(); itr != vec.end(); itr++) {
        if (itr != vec.begin()) {
            cout << ", ";
        }
        cout << *itr;
    }
    cout << ']';
    if (is_newline) {
        cout << endl;
    }
}

template <typename _CharT>
void print2dVec(const vector<vector<_CharT>>& vec) {
    cout << "[";
    for (auto itr = vec.begin(); itr != vec.end(); itr++) {
        if (itr != vec.begin()) {
            cout << endl;
        }
        printVec(*itr, false);
    }
    cout << "]" << endl;
}

template <typename _CharT>
void printArr(_CharT mat, int M) {
    cout << '[';
    for (int i = 0; i < M; i++) {
        if (i != 0) {
            cout << ", ";
        }
        cout << *(mat + i);
    }
    cout << ']';
    cout << endl;
}

template <typename _CharT>
void printArr(_CharT mat, int s, int e) {
    cout << '[';
    for (int i = s; i < e; i++) {
        if (i != s) {
            cout << ", ";
        }
        cout << *(mat + i);
    }
    cout << ']';
    cout << endl;
}

template <typename _CharT>
void print2dArr(_CharT mat, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (j != 0) {
                cout << ", ";
            }
            cout << *(*(mat + i) + j);
        }
        cout << endl;
    }
}

template <typename _CharT>
_CharT minArr(_CharT* arr, int n) {
    assert(n > 0);
    _CharT output = arr[0];

    for (int i = 1; i < n; ++i) {
        output = MIN(output, arr[i]);
    }
    return output;
}

template <typename _CharT>
_CharT minArr(_CharT* arr, int s, int e) {
    assert(e > s);
    _CharT output = arr[s];

    for (int i = s + 1; i < e; ++i) {
        output = MIN(output, arr[i]);
    }
    return output;
}
}  // namespace util

#endif