from collections import defaultdict
import pandas as pd
import random
from anytree import Node, NodeMixin, LevelOrderIter, RenderTree
import numpy as np

# This is the maximum size of the prefix, suffix and substring that will be counted
MAX_STR_SIZE = 8


# Class to denote the node for a generic summary data structure.
class SummaryDSNode(NodeMixin):
    def __init__(self, name, parent=None, children=None):
        super(SummaryDSNode, self).__init__()
        self.name = name
        self.frequency = 1
        self.parent = parent
        self.char_to_children_dict = {}
        self.transition_probabilities = {}

    # Compute transition probabilities based on Eq 5 of the paper
    def update_transition_probabilities(self, root_node):
        k = len(self.children)
        total_frequency = sum([child.frequency for child in self.children])
        numerator, denominator = k, k+1

        if self.parent == root_node:
            numerator = k + 1
        else:
            self.transition_probabilities[self.parent] = 1.0 / denominator
        fraction = (numerator / denominator)
        for child in self.children:
            probability = 0.0
            if total_frequency > 0:
                probability = (child.frequency / total_frequency) * fraction
            self.transition_probabilities[child] = probability

    def count_nodes_same_frequency(self):
        count = 1
        stack_list = [self]
        while len(stack_list) > 0:
            cur_node: SummaryDSNode = stack_list.pop()
            for child_node in cur_node.char_to_children_dict.values():
                child_node: SummaryDSNode
                if child_node.frequency == self.frequency:
                    count += 1
                    stack_list.append(child_node)
        return count

    def count_nodes(self):
        count = 1
        stack_list = [self]
        while len(stack_list) > 0:
            cur_node: SummaryDSNode = stack_list.pop()
            for child_node in cur_node.char_to_children_dict.values():
                child_node: SummaryDSNode
                count += 1
                stack_list.append(child_node)
        return count

    def prune_empty(self):
        stack_list = [self]
        while len(stack_list) > 0:
            cur_node: SummaryDSNode = stack_list.pop()
            prune_ch_list = []
            for ch, child_node in cur_node.char_to_children_dict.items():
                child_node: SummaryDSNode
                if child_node.frequency <= 1:
                    prune_ch_list.append(ch)
                stack_list.append(child_node)
            for ch in prune_ch_list:
                child_node = cur_node.char_to_children_dict[ch]
                child_node.parent = None
                del cur_node.char_to_children_dict[ch]

    def checkout(self):
        stack_list = [self]
        while len(stack_list) > 0:
            cur_node: SummaryDSNode = stack_list.pop()
            for child_node in cur_node.char_to_children_dict.values():
                child_node: SummaryDSNode
                if not cur_node.is_root:
                    assert child_node.frequency <= cur_node.frequency, (
                        child_node.frequency, cur_node.frequency, child_node.name, cur_node.name)
                stack_list.append(child_node)


# This class represents the entire generic summary data structure.
# Using a common for ease of coding.
# It can be replaced with more performant ones such as prefix trees, suffix trees etc.


class SummaryDataStructure:
    # string_generator_fn is a function that takes a string as input
    # and outputs a list of "substrings" of interest.
    # for e.g. all prefixes, suffixes,
    # max_str_size: will be the largest prefix, substring, suffix string that will be created
    # split_words: whether to ignore spaces in a string.
    # if split_words is true, then "a b" will be inserted as two words a b .. else one word with space.
    def __init__(self, string_generator_fn, max_str_size=None, split_words=True):
        if max_str_size is None:
            global MAX_STR_SIZE
            max_str_size = MAX_STR_SIZE
        self.string_generator_fn = string_generator_fn
        self.max_str_size = max_str_size
        self.split_words = split_words
        self.root_node = SummaryDSNode('')

    def insert_string(self, string):
        substrings_of_interest = self.string_generator_fn(string)
        for substring in substrings_of_interest:
            cur_node = self.root_node
            for index, char in enumerate(substring):
                if char in cur_node.char_to_children_dict:
                    cur_node = cur_node.char_to_children_dict[char]
                else:
                    new_node = SummaryDSNode(
                        substring[:index+1], parent=cur_node)
                    cur_node.char_to_children_dict[char] = new_node
                    cur_node = new_node
            # Increment the frequency of the last node
            cur_node.frequency = cur_node.frequency + 1

    def set_frequency(self, string, frequency):
        global MAX_STR_SIZE
        if len(string) <= MAX_STR_SIZE:
            cur_node = self.root_node
            for index, char in enumerate(string):
                assert char in cur_node.char_to_children_dict, (index, char, string, frequency)
                cur_node = cur_node.char_to_children_dict[char]
            cur_node.frequency = frequency

    def search(self, string):
        cur_node = self.root_node
        for index, char in enumerate(string):
            if char in cur_node.char_to_children_dict:
                cur_node = cur_node.char_to_children_dict[char]
            else:
                return None
        return cur_node

    def update_summary_ds_from_file(self, input_file_name, max_n=None):
        with open(input_file_name) as f:
            for i, line in enumerate(f):
                strings = [line.strip()]
                if self.split_words:
                    strings = line.strip().split()
                for string in strings:
                    self.insert_string(string)
                if max_n is not None:
                    if i >= max_n:
                        break

    # returns a data frame with all the strings in the summary data structure and its frequencies
    def get_selectivities(self):
        string_frequency_dict = defaultdict(int)
        for node in LevelOrderIter(self.root_node):
            if node.is_root == False:
                string_frequency_dict[node.name] = max(0, node.frequency - 1)
        df = pd.DataFrame.from_dict(string_frequency_dict, orient='index')
        df.index.name = "string"
        df.columns = ["selectivity"]
        return df

    def update_transition_probabilities(self):
        for node in LevelOrderIter(self.root_node):
            if node.is_root == False:
                node.update_transition_probabilities(self.root_node)

    # For each node,
    #   get the transition probabilities of going to other nodes
    #   use it to get num_triplets_per_node positive random samples using weighted sampling
    #   get num_triplets_per_node random strings as negative samples
    def get_triplets(self, random_seed=1234, num_triplets_per_node=4):
        random.seed(random_seed)
        self.update_transition_probabilities()

        # Get all the strings - it is needed to get dissimilar strings
        all_strings = [node.name for node in LevelOrderIter(
            self.root_node) if not node.is_root]
        total_nodes = len(all_strings)

        all_triplets = []

        for node in LevelOrderIter(self.root_node):
            # The root node is ornamental!
            if node.is_root:
                continue

            candidate_nodes = []
            candidate_probabilities = []

            # get all the neighbors of this node
            for other_node in node.transition_probabilities.keys():
                candidate_nodes.append(other_node.name)
                probability = node.transition_probabilities[other_node]
                candidate_probabilities.append(probability)

            for other_node in node.transition_probabilities.keys():
                for other_other_node in other_node.transition_probabilities.keys():
                    candidate_nodes.append(other_other_node.name)
                    # probability of reaching other_other_node from node
                    new_probability = probability * \
                        other_node.transition_probabilities[other_other_node]
                    candidate_probabilities.append(new_probability)

            if len(candidate_nodes) == 0:
                negatives = random.choices(
                    population=all_strings, k=num_triplets_per_node)
                anchor = node.name
                for index in range(num_triplets_per_node):
                    all_triplets.append((anchor, anchor, negatives[index]))
                continue

            # normalize probabilities if needed
            candidate_probabilities_sum = sum(candidate_probabilities)
            candidate_probabilities = [
                elem/candidate_probabilities_sum for elem in candidate_probabilities]

            # Do a weighted random sampling of to get #num_triplets_per_node nodes
            # from candidates based num_triplets_per_node
            candidate_probabilities = list(candidate_probabilities)
            positives = random.choices(
                population=candidate_nodes, k=num_triplets_per_node, weights=candidate_probabilities)
            negatives = random.choices(
                population=all_strings, k=num_triplets_per_node)
            anchor = node.name
            for index in range(num_triplets_per_node):
                all_triplets.append(
                    (anchor, positives[index], negatives[index]))

        df = pd.DataFrame(all_triplets, columns=[
                          "Anchor", "Positive", "Negative"])
        return df

    def prune(self, pr_lvl, seed):
        n_total_nodes = self.root_node.count_nodes()
        selectivities = self.get_selectivities()
        n_selectivities = len(selectivities)
        selectivities['selectivity'] += 1
        n_remain = int(pr_lvl * n_selectivities)
        assert n_remain > 0, f"too small prune level {pr_lvl}"
        selectivities_sorted = sorted(selectivities['selectivity'], reverse=True)
        pr_thrs = selectivities_sorted[n_remain - 1]
        remain_df_dn = selectivities[selectivities['selectivity'] >= pr_thrs]
        remain_df_up = selectivities[selectivities['selectivity'] > pr_thrs]
        max_num = len(remain_df_dn)
        min_num = len(remain_df_up)
        assert max_num >= n_remain
        assert min_num <= n_remain
        ccccc = len(selectivities[selectivities['selectivity'] == 1])

        if max_num >= n_remain:
            print("prune some nodes with a probability")
            n_remain_same = max_num - min_num  # number of nodes whose values equal pr_thrs
            n_remain_prune = max_num - n_remain
            # pr_rate = (max_num - n_remain) / (max_num - min_num)
            pr_rate = n_remain_prune / n_remain_same
            np.random.seed(seed)
            # pr_map = np.zeros(max_num - min_num, dtype=int)
            # pr_map[:(max_num-n_remain)] = 1
            # np.random.shuffle(pr_map)

        n_added_nodes = 1
        n_pruned_same_nodes_we_want = max_num - n_remain
        n_pruned_nodes_we_want = n_selectivities - n_remain
        n_pruned_nodes = 0
        stack_list = []
        stack_list.append(self.root_node)
        while len(stack_list) > 0:
            cur_node: SummaryDSNode = stack_list.pop()
            prune_char_list = []
            for char, child_node in cur_node.char_to_children_dict.items():
                child_node: SummaryDSNode
                assert child_node.frequency >= 1
                if child_node.frequency > pr_thrs:
                    stack_list.append(child_node)
                    n_added_nodes += 1
                elif child_node.frequency == pr_thrs:
                    same_count = child_node.count_nodes_same_frequency()
                    if n_remain_prune == 0:
                        stack_list.append(child_node)
                        n_added_nodes += 1
                        n_remain_same -= 1
                        continue
                    # if same_count == 1:
                    #     # if n_remain_prune == n_remain_same:
                    #     #     pr_rate = 1.0
                    #     # else:
                    #     #     pr_rate = n_remain_prune / n_remain_same
                    #     pr_rate = n_remain_prune / n_remain_same
                    # else:
                    #     if same_count > n_remain_prune:
                    #         pr_rate = 0
                    #     else:
                    #         pr_rate = n_remain_prune / same_count / n_remain_same

                    if same_count > n_remain_prune:
                        pr_rate = 0.0
                    elif n_remain_prune == n_remain_same:
                        pr_rate = 1.0
                    else:
                        pr_rate = n_remain_prune / n_remain_same / same_count
                        # assert n_remain_prune <= n_remain_same

                    is_prune = np.random.binomial(size=1, n=1, p=pr_rate)[0]
                    if pr_rate > 0.0:
                        is_prune = True
                    if is_prune:  # should prune
                        prune_char_list.append(char)
                        n_remain_prune -= same_count
                        n_remain_same -= same_count
                        n_pruned_nodes += child_node.count_nodes()
                    else:
                        stack_list.append(child_node)
                        n_added_nodes += 1
                        n_remain_same -= 1
                else:
                    prune_char_list.append(char)
                    n_pruned_nodes += child_node.count_nodes()
            for char in prune_char_list:
                child_node = cur_node.char_to_children_dict[char]
                child_node.parent = None
                del cur_node.char_to_children_dict[char]

        # after prune check
        # self.print_tree()
        selectivities = self.get_selectivities()
        n_nodes = len(selectivities)
        assert n_nodes == n_remain
        if max_num == n_remain:
            assert n_nodes == n_remain, f"The number of nodes {n_nodes} is not n_remain {n_remain}"
        else:
            assert n_nodes >= min_num
            assert n_nodes <= max_num
            assert n_pruned_nodes_we_want == n_pruned_nodes

    def print_tree(self):
        for pre, fill, node in RenderTree(self.root_node):
            print("%s%s:%d" % (pre, node.name, node.frequency))


def get_all_prefixes(string, max_size=None):
    if max_size is None:
        global MAX_STR_SIZE
        max_size = MAX_STR_SIZE

    return [string[:j] for j in range(1, min(max_size, len(string)) + 1)]


def get_all_suffixes(string, max_size=None):
    if max_size is None:
        global MAX_STR_SIZE
        max_size = MAX_STR_SIZE
    return [string[-j:] for j in range(1, min(max_size, len(string)) + 1)]


def get_all_substrings(string, max_size=None):
    if max_size is None:
        global MAX_STR_SIZE
        max_size = MAX_STR_SIZE
    arr = []
    n = len(string)
    for i in range(0, n):
        for j in range(i, n):
            if (j+1 - i) <= max_size:
                arr.append(string[i:(j+1)])
    return arr

# Naive way to compute all strings of interest that avoids the use of summary data structures


def aggregate_strings_of_interest(input_file_name, string_agg_fn,
                                  max_size=MAX_STR_SIZE, split_words=True, output_file_name=None):
    string_frequency_dict = defaultdict(int)

    with open(input_file_name) as f:
        for line in f:
            words = [line.strip()]
            if split_words:
                words = line.strip().split()
            for word in words:
                strings = string_agg_fn(word, max_size)
                for string in strings:
                    string_frequency_dict[string] += 1
    df = pd.DataFrame.from_dict(string_frequency_dict, orient='index')
    df.index.name = "string"
    df.columns = ["selectivity"]

    df = df.sort_values(by="string")
    if output_file_name is not None:
        df.to_csv(output_file_name, index=True, header=True)
    return df, string_frequency_dict


def create_summary_datastructure(input_file_name, string_generator_fn, split_words, max_n=None):
    tree = SummaryDataStructure(string_generator_fn, split_words=split_words)
    tree.update_summary_ds_from_file(input_file_name, max_n)
    return tree


def store_selectivities(tree, output_file_name):
    df = tree.get_selectivities()
    df = df.sort_values(by="string")
    # print("store_selectivities at", output_file_name)
    df.to_csv(output_file_name, index=True, header=True)


def store_triplets(tree, output_file_name):
    df = tree.get_triplets()
    # print("store_triplets at", output_file_name)
    df.to_csv(output_file_name, index=False, header=True)
