import xml.etree.ElementTree as ET
import re
import pandas as pd
import summary_data_structures
import csv
import os

# Matches alphanumeric and space
regex_pattern = r'[^A-Za-z0-9 ]+'


def prepare_dataset(folder_path, train_qs_prefix, total_prefix, file_name_prefix, triplet_name_prefix, threshold):
    # print("Processing data", train_qs_prefix)
    # functions = [summary_data_structures.get_all_prefixes, summary_data_structures.get_all_suffixes,
    #              summary_data_structures.get_all_substrings]
    functions = [summary_data_structures.get_all_prefixes]
    input_qs_file_name = folder_path + train_qs_prefix + ".csv"
    for index, fn in enumerate(functions):
        count_file_name = folder_path + total_prefix + "_counts.csv"
        count_file_name_w = folder_path + file_name_prefix + "_counts.csv"
        triplet_file_name = folder_path + triplet_name_prefix + "_triplets.csv"
        # print("input  qs:", input_qs_file_name)
        # print("total  qs:", count_file_name)
        # print("save   qs:", count_file_name_w)
        # print("save trpl:", triplet_file_name)
        assert os.path.exists(count_file_name), count_file_name
        if os.path.exists(count_file_name_w) and os.path.exists(triplet_file_name):
            # print("Files already exist at", count_file_name_w, "and", triplet_file_name)
            continue

        summary_data_structures.MAX_STR_SIZE = 20
        max_n = None
        tree = summary_data_structures.create_summary_datastructure(
            input_qs_file_name, fn, split_words=False, max_n=max_n)

        prfxes = set()
        with open(input_qs_file_name) as f:
            for i, line in enumerate(f):
                line = line.rstrip()
                for pos in range(len(line)):
                    prfx = line[:pos+1]
                    prfxes.add(prfx)
                if max_n is not None:
                    if i >= max_n:
                        break
        max_nprfx = len(prfxes)

        # tree.print_tree()
        if threshold >= 0:
            with open(count_file_name) as f:
                fr = csv.reader(f, delimiter=',')
                for i, line in enumerate(fr):
                    if i > 0:
                        string, frequency = line[0], int(line[threshold+1])
                        if string in prfxes:
                            tree.set_frequency(string, frequency+1)
                        # if frequency > 0:
                        #     # print(string, frequency)
                        #     tree.set_frequency(string, frequency+1)
                        # else:
                        #     empty_node = tree.search(string)
                        #     if empty_node is not None:
                        #         assert string == empty_node.name
                        #         assert empty_node.parent.char_to_children_dict[string[-1]] is empty_node
                        #         del empty_node.parent.char_to_children_dict[string[-1]]
                        #         empty_node.parent = None

                    if i >= max_nprfx:
                        break
            summary_data_structures.store_selectivities(tree, count_file_name_w)

            # tree.root_node.checkout()
            tree.root_node.prune_empty()
            # if type(prune_level) == float:
            #     tree.prune(prune_level, seed=0)
            # summary_data_structures.store_selectivities(tree, count_file_name_w)
        else:
            summary_data_structures.store_selectivities(tree, count_file_name_w)
        # tree.print_tree()
        summary_data_structures.store_triplets(tree, triplet_file_name)
    # print('\n\n')

# The following functions generates the frequencies and triplets
# This function might take few minutes for large datasets :)
