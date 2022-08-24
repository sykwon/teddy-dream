#ifndef wSED_FOR_CPP_2675DCD0_9480_4c0c_B92A_CC14C027B731
#define wSED_FOR_CPP_2675DCD0_9480_4c0c_B92A_CC14C027B731
// #define __DEBUG__

#include <codecvt>
#include <locale>
#include <vector>

#include "utf8.h"
#include "util.h"

// #define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
// #define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

using namespace std;
namespace wSED {
wstring_convert<codecvt_utf8<char32_t>, char32_t> conv_string2code;
bool utf8compare(const string& str1, const string& str2) {
    auto first1 = str1.begin();
    auto first2 = str2.begin();
    auto last1 = str1.end();
    auto last2 = str2.end();

    while (first1 != last1 && first2 != last2) {
        int code1 = utf8::next(first1, last1);
        int code2 = utf8::next(first2, last2);
        if (code1 != code2) {
            return code1 < code2;
        }
    }
    return first1 == last1;
}

int utf8length(const string& str) {
    return utf8::distance(str.begin(), str.end());
}

vector<uint32_t> string2codes(const string& x) {
    vector<uint32_t> output;
    auto first = x.begin();
    auto last = x.end();
    uint32_t code;
    while (first != last) {
        code = utf8::next(first, last);
        output.push_back(code);
    }
    return output;
}

string codes2string(uint32_t* codes, int len) {
    stringstream ss;
    uint32_t code;
    for (int i = 0; i < len; i++) {
        code = codes[i];
        ss << conv_string2code.to_bytes(code);
    }
    return ss.str();
}

string codes2string(const vector<uint32_t>& codes) {
    stringstream ss;
    for (uint32_t code : codes) {
        ss << conv_string2code.to_bytes(code);
    }
    return ss.str();
}

string u32string2string(const u32string& str) {
    vector<uint32_t> codes(str.size());
    for (int i = 0; i < (int)str.size(); ++i) {
        codes[i] = str[i];
    }
    return codes2string(codes);
}

string anystr2string(const u32string& str) {
    return u32string2string(str);
}

string anystr2string(const string& str) {
    return str;
}

template <typename _CharT>
vector<basic_string<_CharT>> partition_string(const basic_string<_CharT>& r, int n_part) {
    vector<basic_string<_CharT>> output;
    int length = r.size();
    int l_part = length / n_part;
    int remain = length - l_part * n_part;
    // assert(length >= n_part);
    if (length < n_part) {
        assert(false);
        // output.push_back(NULL);
        return output;
    }
    int s = 0;
    int e = 0;
    for (int i = 0; i < n_part; i++) {
        s = e;
        e += l_part;
        if (i < remain) {
            e += 1;
        }
        auto split = r.substr(s, e - s);

#ifdef __DEBUG__
        cout << "s: " << s << " e: " << e << " e-s: " << e - s << " size: " << split.size() << " length: " << length << " n_part " << n_part << endl;
        assert(split.size() > 0);
#endif
        output.push_back(split);
    }
    assert(e == length);
    return output;
}

int SED(const u32string& r, const u32string& s) {
    int r_len = r.size();
    int s_len = s.size();

    int* mem_prev = new int[r_len + 1];
    int* mem = new int[r_len + 1];
    int* tmp_ptr;
    int sed = r_len;

    for (int i = 0; i <= r_len; i++) {
        mem_prev[i] = i;
    }

    for (int j = 1; j <= s_len; j++) {
        mem[0] = 0;
        for (int i = 1; i <= r_len; i++) {
            if (r[i - 1] == s[j - 1]) {
                mem[i] = mem_prev[i - 1];
            } else {
                mem[i] = MIN(mem_prev[i], mem_prev[i - 1]);
                mem[i] = MIN(mem[i], mem[i - 1]) + 1;
            }
        }
        sed = MIN(sed, mem[r_len]);

        tmp_ptr = mem_prev;
        mem_prev = mem;
        mem = tmp_ptr;
    }

    delete[] mem_prev;
    delete[] mem;

    return sed;
}

int SED_intv(const u32string& r, const u32string& s, int delta) {
    /* 
        Only when SED is less than or equal to delta, it returns true SED value.
        If the distance computation is aborted, it returns INT32_MAX.
        Otherwise, it returns distance value which is larger than delta and not correct.
    */
    int r_len = r.size();
    int s_len = s.size();
    int js;
    int je;
    int sed = INT32_MAX;

    int** mem;
    mem = new int*[r_len + 1];
    for (int i = 0; i < r_len + 1; ++i) {
        mem[i] = new int[s_len + 2];
        mem[i][0] = i;
        for (int j = 1; j < s_len + 2; ++j) {
            mem[i][j] = 0;
        }
    }

    js = 1;
    je = s_len + 1;
    for (int i = 1; i < r_len + 1; ++i) {
        for (int j = js; j < je; ++j) {
            if (r[i - 1] == s[j - 1]) {  // i and j are 1-index
                mem[i][j] = mem[i - 1][j - 1];
            } else {
                mem[i][j] = MIN(mem[i - 1][j], mem[i - 1][j - 1]);  // faster than min funtion
                mem[i][j] = MIN(mem[i][j], mem[i][j - 1]) + 1;      // faster than min funtion
            }
        }
        if (i == r_len) {
            sed = util::minArr(mem[r_len], js, je);
        }
    }

    for (int i = 0; i < r_len + 1; ++i) {
        delete[] mem[i];
    }
    delete[] mem;

    return sed;
}

}  // namespace wSED
#endif  // header guard