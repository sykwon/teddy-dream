#ifndef trie_FOR_CPP_2675DCD0_9480_4c0c_B92A_CC14C027B731
#define trie_FOR_CPP_2675DCD0_9480_4c0c_B92A_CC14C027B731

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "util.h"
#include "wSED.h"

#define TRIE_HASH_SIZE 30

using namespace std;

int __rid__;

template <typename _CharT, typename _ValT>
class TrieStoreNode {
public:
    _CharT chr;
    _ValT val = nullptr;
    unordered_map<_CharT, TrieStoreNode*> children;
    TrieStoreNode* parent;
    // int depth = 0;
    // int rid = -1;
    // vector<int> selectivities;
    TrieStoreNode(_CharT chr = '\0', TrieStoreNode* parent = nullptr) : chr(chr), parent(parent) {
        // if (parent) {
        //     this->depth = parent->depth + 1;
        // }
    }

    ~TrieStoreNode() {
        if (!is_leaf()) {
            for (auto pair : children) {
                auto child = pair.second;
                delete child;
            }
        }
        delete val;
    }

    bool is_root() {
        return !(this->parent);
    }

    bool is_leaf() {
        return this->children.size() == 0;
    }

    // bool is_active() {
    //     return this->rid >= 0;
    // }

    basic_string<_CharT> node_string() {
        basic_stringstream<_CharT> output;
        TrieStoreNode* curr_node = this;
        cout << "node: ";
        while (!curr_node->is_root()) {
            // cout<< curr_node->chr << ", ";
            output << curr_node->chr;
            curr_node = curr_node->parent;
        }
        basic_string<_CharT> outstr = output.str();
        reverse(outstr.begin(), outstr.end());
        return outstr;
    }

    TrieStoreNode* search(basic_string<_CharT> query) {
        auto curr_node = this;
        if (query.size() == 0) {
            return curr_node;
        }

        int p = 0;
        while (p < query.size()) {
            auto chr = query[p];
            if (curr_node->children.find(chr) != curr_node->children.end()) {
                curr_node = curr_node->children[chr];
            } else {
                return nullptr;
            }
            ++p;
        }
        return curr_node;
    }

    TrieStoreNode* add_char(_CharT chr) {
        // string chr = conv_string2code.to_bytes(chr);
        TrieStoreNode* child;
        if (this->children.find(chr) != this->children.end()) {
            child = this->children[chr];
        } else {
            child = new TrieStoreNode(chr, this);
            this->children[chr] = child;
        }
        return child;
    }

    void print_curr_node() {
        // printf("rid: %d, str: %s\n", this->rid, this->node_string().c_str());
        printf("str: %s\n", this->node_string().c_str());
    }

    // vector<string> sub_trie_string() {
    //     vector<string> strings;
    //     stringstream ss;
    //     stringstream prefix_ss;
    //     if (this->is_leaf()) {
    //         // string str;
    //         // str.push_back(this->chr());
    //         ss << this->chr();
    //         // strings.push_back(str);
    //         strings.push_back(ss.str());
    //     } else {
    //         for (auto itr = this->children.begin(); itr != this->children.end(); itr++) {
    //             auto kv = *itr;
    //             // char key = get<0>(kv);
    //             TrieStoreNode* child = get<1>(kv);
    //             vector<string> child_strings = child->sub_trie_string();
    //             ss.str("");
    //             // string start_prefix = "";
    //             if (itr == this->children.begin()) {
    //                 // start_prefix.push_back(this->chr());
    //                 // start_prefix.push_back('-');
    //                 ss << this->chr();
    //                 ss << '-';
    //             } else {
    //                 // start_prefix = " ˪";
    //                 ss << " ˪";
    //             }
    //             string start_prefix = ss.str();

    //             string connect_prefix;
    //             if (next(itr) == this->children.end()) {
    //                 connect_prefix = "  ";
    //             } else {
    //                 connect_prefix = " |";
    //             }

    //             for (auto itr = child_strings.begin(); itr != child_strings.end(); itr++) {
    //                 string prefix;
    //                 string child_string = *itr;
    //                 if (itr == child_strings.begin()) {
    //                     prefix = start_prefix;
    //                 } else {
    //                     prefix = connect_prefix;
    //                 }

    //                 if (this->is_root()) {
    //                     prefix.clear();
    //                     prefix.push_back(prefix.back());
    //                 }
    //                 cout << "record :" << prefix + child_string << endl;
    //                 strings.push_back(prefix + child_string);
    //             }
    //         }
    //     }
    //     return strings;
    // }

    // void print_sub_trie() {
    //     for (string str : this->sub_trie_string()) {
    //         cout << str << endl;
    //     }
    // }
};

template <typename _CharT, typename _ValT>
class TrieStore {
    typedef TrieStoreNode<_CharT, _ValT> _nodeT;
    typedef basic_string<_CharT> _stringT;

public:
    _nodeT* root = nullptr;

    TrieStore<_CharT, _ValT>() {
        root = new _nodeT();
    }

    ~TrieStore<_CharT, _ValT>() {
        delete root;
    }

    _nodeT* search(_stringT& rec) {
        return root->search(rec);
    }

    unordered_set<int> candidate(_stringT& rec) {
        unordered_set<int> output;
        int s_len = rec.size();
        int curr_pos;
        _nodeT* curr_node;
        _CharT curr_chr;

        curr_node = this->root;
        if (curr_node && curr_node->val) {
            for (int cand : *(curr_node->val)) {
                output.insert(cand);
            }
        }

        for (int i = 0; i < s_len; ++i) {
            curr_pos = i;
            curr_node = this->root;

            while (curr_node && curr_pos < s_len) {
                curr_chr = rec[curr_pos];
                curr_pos += 1;

                if (curr_node->children.find(curr_chr) != curr_node->children.end()) {
                    curr_node = curr_node->children[curr_chr];
                } else {
                    curr_node = nullptr;
                }

                if (curr_node && curr_node->val) {
                    for (int cand : *(curr_node->val)) {
                        output.insert(cand);
                    }
                }
            }
        }
        return output;
    }

    _nodeT* add_key_val(_stringT& key, _ValT val) {
        _nodeT* curr_node = this->root;
        for (auto chr : key) {
            curr_node = curr_node->add_char(chr);
        }
#ifdef JOIN_DEBUG
        if (key.size() == 0) {
            cout << "empty" << endl;
            if (curr_node != this->root) {
                assert(false);
            }
        }
#endif
        curr_node->val = val;
        return curr_node;
    }

    void add_kv_pairs(vector<_stringT>& keys, vector<_ValT>& vals) {
        for (int i = 0; i < (int)keys.size(); i++) {
            add_key_val(keys[i], vals[i]);
        }
    }
};

template <typename _CharT = char32_t>
class Trie;

template <typename _CharT = char32_t>
class BaseNode;

template <typename _CharT>
class BaseNode {
public:
    int depth;
    int rid;
    _CharT chr;

    BaseNode() {}
    ~BaseNode() {}

    void set_active_node(int rid) {
        this->rid = rid;
    }

    virtual bool is_root() = 0;

    virtual bool is_leaf() = 0;

    bool is_active() {
        return this->rid >= 0;
    }

    // virtual basic_string<_CharT> node_string() = 0;

    // void print_curr_node() {
    //     string node_str = wSED::anystr2string(this->node_string());
    //     printf("rid: %02d, str: %s\n", this->rid, node_str.c_str());
    // }
};

template <typename _CharT = char32_t>
class TrieNode;
template <typename _CharT>
class TrieNode : public BaseNode<_CharT> {
public:
    TrieNode<_CharT>* parent;
    unordered_map<_CharT, TrieNode<_CharT>*> children;
    TrieNode<_CharT>() {
        children.reserve(TRIE_HASH_SIZE);
        this->rid = -1;
    }
    ~TrieNode<_CharT>() {
    }
    void setNode(_CharT chr, TrieNode<_CharT>* parent) {
        this->chr = chr;
        this->parent = parent;
        this->depth = parent->depth + 1;
    }

    bool is_root() {
        return !(this->parent);
    }

    bool is_leaf() {
        return this->children.size() == 0;
    }
    TrieNode<_CharT>* add_char(_CharT chr) {
        if (this->children.find(chr) == this->children.end()) {
            Trie<_CharT>::last_pos += 1;
            auto child_node = &Trie<_CharT>::node_arr[Trie<_CharT>::last_pos];
            this->children[chr] = child_node;
            child_node->chr = chr;
            child_node->parent = this;
            child_node->depth = this->depth + 1;
        }
        return this->children[chr];
    }
    TrieNode<_CharT>* add_char_with_id(_CharT chr) {
        if (this->children.find(chr) == this->children.end()) {
            Trie<_CharT>::last_pos += 1;
            auto child_node = &Trie<_CharT>::node_arr[Trie<_CharT>::last_pos];
            this->children[chr] = child_node;
            child_node->chr = chr;
            child_node->parent = this;
            child_node->depth = this->depth + 1;
            child_node->rid = __rid__;
            __rid__ += 1;
        }
        return this->children[chr];
    }

    basic_string<_CharT> node_string() {
        basic_stringstream<_CharT> output;
        auto curr_node = this;
        while (curr_node->parent) {
            output << curr_node->chr;
            curr_node = curr_node->parent;
        }
        basic_string<_CharT> outstr = output.str();
        reverse(outstr.begin(), outstr.end());
        return outstr;
    }

    long int size() {
        long int count = 1;  // self
        for (auto kv : children) {
            auto child = get<1>(kv);
            count += child->size();
        }
        return count;
    }

    void print_tree(bool first = true, bool is_rid = false) {
        if (!first) {
            for (int i = 0; i < this->depth - 1; ++i) {
                if (i != this->depth - 2) {
                    cout << ' ';
                } else {
                    cout << '-';
                }
            }
            // cout << "depth: " << this->depth << endl;
        }
        char ch = this->chr;
        if (!is_rid) {
            cout << ch;

        } else {
            cout << ' ' << this->rid;
        }
        bool is_first = true;
        vector<char> keys;
        for (auto kv : children) {
            keys.push_back(get<0>(kv));
            // wcout << (wchar_t)get<0>(kv) << endl;
        }
        sort(keys.begin(), keys.end());
        // sort(children.begin(), children.end());

        // for (auto kv : children) {
        for (auto ch : keys) {
            // char ch = get<0>(kv);
            // TrieNode<_CharT>* child_node = get<1>(kv);
            TrieNode<_CharT>* child_node = children[ch];
            // cout << "depth: " << child_node->depth << endl;
            // for (int i = 0; i < child_node->depth; ++i) {
            //     cout << "-";
            // }
            child_node->print_tree(is_first, is_rid);
            is_first = false;
        }
        if (is_leaf()) {
            cout << "$" << endl;
        }
    }
};

template <typename _CharT = char32_t>
class RadixTree;

template <typename _CharT = char32_t>
class RadixTreeNode {
    // typedef basic_stringstream<_CharT> _ssT;
    typedef basic_string<_CharT> _sT;
    typedef RadixTreeNode<_CharT> _nodeT;
    typedef RadixTree<_CharT> _trieT;

public:
    _CharT* s_str;
    int* s_rid;
    _nodeT** children = nullptr;
    // _nodeT* sibling = nullptr;
    _nodeT* next_node = nullptr;
    int depth;
    int n_str;
    // int n_child = 0;

    RadixTreeNode<_CharT>() {}
    ~RadixTreeNode<_CharT>() {}

    void set_subtrie(TrieNode<_CharT>* src_root) {
        int n_child = 0;
        // set children space
        this->children = &_trieT::child_ptr_arr[_trieT::child_last_pos + 1];
        // this->children = &_trieT::node_arr[_trieT::node_last_pos + 1];
        _trieT::child_last_pos += src_root->children.size() + 1;
        vector<_CharT> keys;
        for (auto kv : src_root->children) {
            keys.push_back(get<0>(kv));
        }
        sort(keys.begin(), keys.end());

        // _nodeT* prev_node = nullptr;
        // for (auto kv : src_root->children) {
        for (auto key : keys) {
            _trieT::str_last_pos += 1;

            // TrieNode<_CharT>* src_child = get<1>(kv);
            TrieNode<_CharT>* src_child = src_root->children[key];

            // set ch & rid
            // _trieT::chr_arr[_trieT::str_last_pos] = get<0>(kv);
            _trieT::chr_arr[_trieT::str_last_pos] = key;
            _trieT::rid_arr[_trieT::str_last_pos] = src_child->rid;

            _trieT::node_last_pos += 1;

            // setting cur child node
            _nodeT* dest_child = &_trieT::node_arr[_trieT::node_last_pos];
            dest_child->s_str = &_trieT::chr_arr[_trieT::str_last_pos];
            dest_child->s_rid = &_trieT::rid_arr[_trieT::str_last_pos];
            // cout << "build s_str: " << dest_child->s_str << endl;
            dest_child->n_str = 1;

            // if (prev_node) {
            //     prev_node->sibling = dest_child;
            // }
            // prev_node = dest_child;

            // compress single edged nodes
            while (src_child->children.size() == 1) {
                for (auto kv2 : src_child->children) {
                    _trieT::str_last_pos += 1;
                    // set ch & rid
                    // char ch = get<0>(kv2);
                    // cout << "while build ch: " << ch << endl;
                    _trieT::chr_arr[_trieT::str_last_pos] = get<0>(kv2);
                    src_child = get<1>(kv2);
                    _trieT::rid_arr[_trieT::str_last_pos] = src_child->rid;
                }
                dest_child->n_str += 1;
            }
            dest_child->depth = this->depth + MAX(this->n_str, 1);  // MAX for root
            children[n_child] = dest_child;                         // enlist child_ptr
            n_child += 1;
            // assert(src_child->children.size() != 1);  // for debug
            if (src_child->children.size() > 1) {
                dest_child->set_subtrie(src_child);
            } else if (src_child->children.size() == 0) {
                // dest_child
                _trieT::child_last_pos += 1;
                dest_child->children = &_trieT::child_ptr_arr[_trieT::child_last_pos];
                dest_child->children[0] = nullptr;
            }
        }
        children[n_child] = nullptr;
        // assert(n_child == src_root->children.size()); // for debug
        // n_child = src_root->children.size();
    }

    void set_next(_nodeT* next = nullptr) {
        if (next) {
            next_node = next;  // The next_node means the node we need to visit when the current node is pruned.
        }
        if (!this->children[0]) {
            return;
        }
        _nodeT** children = this->children;
        _nodeT* child_node = children[0];
        _nodeT* sibling_node;
        while (child_node) {
            children += 1;
            sibling_node = *children;
            child_node->set_next(sibling_node ? sibling_node : next);
            child_node = sibling_node;
        }
        // for (int i = 0; i < this->n_child; ++i) {
        //     child_node = this->children[i];
        //     if (i + 1 == this->n_child) {
        //         sibling_node = nullptr;
        //     } else {
        //         sibling_node = this->children[i + 1];
        //     }
        //     child_node->set_next(sibling_node ? sibling_node : next);
        //     // if (i + 1 == this->n_child) {
        //     //     child_node->set_next(nullptr);
        //     // } else {
        //     //     child_node->set_next(this->children[i + 1]);
        //     // }
        //     // child_node->set_next(child_node->sibling ? child_node->sibling : next);
        //     // child_node = child_node->sibling;
        // }
    }

    bool is_root() {
        return depth == 0;
    }

    bool is_leaf() {
        // return n_child == 0;
        return !this->children[0];
    }

    _CharT chr(int pos) {
        // assert(pos < n_str);  // for debug
        // return RadixTree<_CharT>::chr_arr[s_str + pos];
        return *(s_str + pos);
    }

    int rid(int pos) {
        // assert(pos < n_str);  // for debug
        // return RadixTree<_CharT>::rid_arr[s_str + pos];
        return *(s_rid + pos);
    }

    bool is_active(int pos) {
        return rid(pos) >= 0;
    }

    void print_tree(bool first = true, bool is_rid = false) {
        // cout << "test" << endl;
        if (!first) {
            for (int i = 0; i < depth - 1; ++i) {
                cout << "-";
            }
        }
        // cout << chr(0) << endl;
        // cout << is_leaf() << endl;
        for (int i = 0; i < n_str; ++i) {
            if (!is_rid) {
                char ch = this->chr(i);  // RadixTree<_CharT>::chr_arr[s_str + i];
                cout << ch;
            } else {
                int rid = this->rid(i);  // RadixTree<_CharT>::rid_arr[s_str + i];
                cout << ' ' << rid;
                cout << " [ " << s_str << " ] ";
            }
        }
        if (is_leaf()) {
            cout << "$" << endl;
        } else {
            // cout << endl;
            // _nodeT* child_node;
            // for (int i = 0; i < n_child; ++i) {
            //     // RadixTreeNode<_CharT>* child_node = *(children + i);
            //     // RadixTreeNode<_CharT>* child_node = children[i];
            //     // _nodeT* child_node = children + i;
            //     child_node = children[i];
            //     child_node->print_tree(i == 0, is_rid);
            //     // child_node = child_node->sibling;
            // }
            _nodeT** child_node = children;
            // wcout << (wchar_t)(*child_node)->chr(0) << endl;
            // if (*child_node != nullptr) {
            //     cout << "not null" << endl;
            // } else {
            //     cout << "null" << endl;
            // }
            bool is_first = true;
            // cout << "while loop before" << endl;
            while (*child_node != nullptr) {
                // cout << "while loop" << endl;
                // wcout << (wchar_t)child_node->chr(0) << endl;
                (*child_node)->print_tree(is_first, is_rid);
                is_first = false;
                child_node = child_node + 1;
            }
        }
    }
};

template <typename _CharT = char32_t>
class TrieList;

template <typename _CharT = char32_t>
class TrieListNode : public BaseNode<_CharT> {
public:
    // TrieListNode<_CharT>* parent;
    // vector<TrieListNode<_CharT>*> children;
    TrieListNode<_CharT>** children;
    int n_child = 0;

    TrieListNode<_CharT>() {}
    ~TrieListNode<_CharT>() {}

    void set_subtrie(TrieNode<_CharT>* src_root) {
        children = &TrieList<_CharT>::child_ptr_arr[TrieList<_CharT>::ptr_last_pos];
        TrieList<_CharT>::ptr_last_pos += src_root->children.size();
        vector<_CharT> keys;
        for (auto kv : src_root->children) {
            keys.push_back(get<0>(kv));
        }
        sort(keys.begin(), keys.end());

        // for (auto kv : src_root->children) {
        for (auto ch : keys) {
            // auto ch = get<0>(kv);
            // auto src_child = get<1>(kv);
            auto src_child = src_root->children[ch];
            TrieList<_CharT>::last_pos += 1;
            TrieListNode<_CharT>* dest_child = &TrieList<_CharT>::node_arr[TrieList<_CharT>::last_pos];
            dest_child->chr = ch;
            dest_child->depth = this->depth + 1;
            // dest_child->parent = this;
            dest_child->rid = src_child->rid;
            // children.push_back(dest_child);
            children[n_child] = dest_child;
            n_child += 1;
            dest_child->set_subtrie(src_child);
        }
        // assert(n_child == src_root->children.size());
    }

    bool is_root() {
        // return !(this->parent);
        return this->chr == '\0';
    }

    bool is_leaf() {
        // return children.size() == 0;
        return n_child == 0;
    }

    // basic_string<_CharT> node_string() {
    //     basic_stringstream<_CharT> output;
    //     auto curr_node = this;
    //     while (curr_node->parent) {
    //         output << curr_node->chr;
    //         curr_node = curr_node->parent;
    //     }
    //     basic_string<_CharT> outstr = output.str();
    //     reverse(outstr.begin(), outstr.end());
    //     return outstr;
    // }

    long int size() {
        long int count = 1;  // self
        // for (auto child : children) {
        //     count += child->size();
        // }
        for (int i = 0; i < n_child; ++i) {
            count += children[i]->size();
        }
        return count;
    }

    int active_count() {
        int count = (int)this->is_active();
        // for (auto child : children) {
        //     count += child->active_count();
        // }
        for (int i = 0; i < n_child; ++i) {
            count += children[i]->active_count();
        }
        return count;
    }

    int single_edge_count() {
        int count = (int)(this->n_child == 1);
        for (int i = 0; i < n_child; ++i) {
            count += children[i]->single_edge_count();
        }
        return count;
    }
};

template <typename _CharT>
class Trie {
    typedef TrieNode<_CharT> _nodeT;
    typedef basic_string<_CharT> _stringT;

public:
    static _nodeT* node_arr;
    static int last_pos;
    _nodeT* root = nullptr;

    Trie<_CharT>() {
    }
    Trie<_CharT>(int n_prfx) {
        node_arr = new _nodeT[n_prfx + 1];
        root = &node_arr[0];
        root->chr = '\0';
        root->depth = 0;
        root->rid = -1;
        last_pos = 0;
    }
    ~Trie<_CharT>() {
        delete[] node_arr;
    }

    _nodeT* search(_stringT& rec) {
        _nodeT* curr_node = root;
        for (_CharT chr : rec) {
            if (curr_node->children.find(chr) != curr_node->children.end()) {
                curr_node = curr_node->children[chr];
            } else {
                return nullptr;
            }
        }
        return curr_node;
    }

    _nodeT* add_string(_stringT& query) {
        _nodeT* curr_node = this->root;
        for (_CharT chr : query) {
            curr_node = curr_node->add_char(chr);
        }
        return curr_node;
    }

    _nodeT* add_all_prefixes(_stringT& query) {
        _nodeT* curr_node = this->root;
        for (_CharT chr : query) {
            curr_node = curr_node->add_char_with_id(chr);
        }
        return curr_node;
    }

    void add_strings(vector<_stringT>& queries) {
        _nodeT* curr_node;
        _stringT query;
        for (int rid = 0; rid < (int)queries.size(); rid++) {
            query = queries[rid];
            // cout<< "node rid" << endl;
            //  cout<< "query: " <<query<<endl;
            curr_node = this->add_string(query);
            curr_node->set_active_node(rid);
            // cout<< "rid: "<<curr_node->rid << "active: " <<curr_node->is_active() << endl;
        }
    }

    int add_all_prefixes(vector<_stringT>& queries) {
        // return the size of trie
        __rid__ = 0;
        // _nodeT* curr_node;
        _stringT query;
        for (int rid = 0; rid < (int)queries.size(); rid++) {
            query = queries[rid];
            // cout<< "node rid" << endl;
            //  cout<< "query: " <<query<<endl;
            this->add_all_prefixes(query);
            // curr_node->set_active_node(rid);
            // cout<< "rid: "<<curr_node->rid << "active: " <<curr_node->is_active() << endl;
        }
        int trie_size = __rid__;
        return trie_size;
    }

    long int size() {
        return root->size();
    }
};

template <typename _CharT>
class RadixTree {
    typedef RadixTreeNode<_CharT> _nodeT;
    typedef basic_string<_CharT> _stringT;

public:
    static _nodeT* node_arr;
    static _nodeT** child_ptr_arr;
    static _CharT* chr_arr;
    static int* rid_arr;
    static int node_last_pos;
    static int child_last_pos;
    static int str_last_pos;
    _nodeT* root;
    RadixTree<_CharT>(Trie<_CharT>* trie) {
        node_arr = new _nodeT[trie->size()];
        child_ptr_arr = new _nodeT*[trie->size() * 2];
        chr_arr = new _CharT[trie->size()];
        rid_arr = new int[trie->size()];

        root = &node_arr[0];
        chr_arr[0] = '\0';
        rid_arr[0] = -1;
        root->s_str = chr_arr;
        root->s_rid = rid_arr;
        root->n_str = 1;
        root->depth = 0;
        node_last_pos = 0;
        child_last_pos = 0;
        str_last_pos = 0;
        root->set_subtrie(trie->root);
    }
    ~RadixTree<_CharT>() {
        delete[] node_arr;
        delete[] child_ptr_arr;
        delete[] chr_arr;
        delete[] rid_arr;
    }
};

template <typename _CharT>
RadixTreeNode<_CharT>* RadixTree<_CharT>::node_arr;
template <typename _CharT>
RadixTreeNode<_CharT>** RadixTree<_CharT>::child_ptr_arr;
template <typename _CharT>
_CharT* RadixTree<_CharT>::chr_arr;
template <typename _CharT>
int* RadixTree<_CharT>::rid_arr;
template <typename _CharT>
int RadixTree<_CharT>::node_last_pos;
template <typename _CharT>
int RadixTree<_CharT>::child_last_pos;
template <typename _CharT>
int RadixTree<_CharT>::str_last_pos;

template <typename _CharT>
class TrieList {
    typedef TrieListNode<_CharT> _nodeT;
    typedef basic_string<_CharT> _stringT;

public:
    static _nodeT* node_arr;
    static _nodeT** child_ptr_arr;
    static int last_pos;
    static int ptr_last_pos;
    _nodeT* root;
    TrieList<_CharT>(Trie<_CharT>* trie) {
        node_arr = new _nodeT[trie->size()];
        child_ptr_arr = new _nodeT*[trie->size() - 1];
        root = &node_arr[0];
        root->chr = '\0';
        root->rid = -1;
        // root->parent = nullptr;
        root->depth = 0;
        last_pos = 0;
        ptr_last_pos = 0;
        root->set_subtrie(trie->root);
    }
    ~TrieList<_CharT>() {
        delete[] node_arr;
    }

    long int size() {
        return root->size();
    }

    void print_node_list() {
        for (int idx = 0; idx <= last_pos; ++idx) {
            _nodeT* node = &this->node_arr[idx];
            node->print_curr_node();
        }
    }
};
template <typename _CharT>
TrieListNode<_CharT>* TrieList<_CharT>::node_arr;
template <typename _CharT>
TrieListNode<_CharT>** TrieList<_CharT>::child_ptr_arr;
template <typename _CharT>
int TrieList<_CharT>::last_pos;
template <typename _CharT>
int TrieList<_CharT>::ptr_last_pos;

template <typename _CharT>
TrieNode<_CharT>* Trie<_CharT>::node_arr;
template <typename _CharT>
int Trie<_CharT>::last_pos;
template <typename _CharT>
Trie<_CharT>* build_trie(vector<basic_string<_CharT>>& R, int delta) {
    Trie<_CharT>* trie = new Trie<_CharT>();
    trie->add_strings(R);
    return trie;
}

#endif
