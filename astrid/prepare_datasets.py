import xml.etree.ElementTree as ET
import re
import pandas as pd
import summary_data_structures
import csv
import os

# Matches alphanumeric and space
regex_pattern = r'[^A-Za-z0-9 ]+'

# Download dblp50000.xml from HPI at
# https://hpi.de/fileadmin/user_upload/fachgebiete/naumann/projekte/repeatability/DBLP/dblp50000.xml


def load_and_save_dblp():
    tree = ET.parse('dblp50000.xml')
    root = tree.getroot()

    titles = []
    authors = []

    for article in root:
        title = article.find("title").text
        titles.append(title)
        for authorNode in article.findall("author"):
            authors.append(authorNode.text)

    titles = pd.Series(titles)
    authors = pd.Series(authors)

    titles.str.replace(regex_pattern, '').to_csv(
        "datasets/dblp/dblp_titles.csv", index=False, header=False)
    authors.str.replace(regex_pattern, '').to_csv(
        "datasets/dblp/dblp_authors.csv", index=False, header=False)

# Download from https://github.com/gregrahn/join-order-benchmark


def load_imdb_movie_titles():
    df = pd.read_csv("title.csv", header=None,
                     warn_bad_lines=True, error_bad_lines=False)
    # second column is the title
    df[1].str.replace(regex_pattern, '').to_csv(
        "datasets/imdb/imdb_movie_titles.csv", index=False, header=False)

# Download from https://github.com/gregrahn/join-order-benchmark


def load_imdb_movie_actors():
    df = pd.read_csv("name.csv", header=None,
                     warn_bad_lines=True, error_bad_lines=False)
    # second column is the title
    df[1].str.replace(regex_pattern, '').to_csv(
        "datasets/imdb/imdb_movie_actors.csv", index=False, header=False)

# download from https://github.com/electrum/tpch-dbgen
# schema diagram at https://docs.deistercloud.com/content/Databases.30/TPCH%20Benchmark.90/Data%20generation%20tool.30.xml?embedded=true


def load_and_save_tpch():
    col_names = ["partkey", "name", "mfgr", "brand", "type",
                 "size", "container", "retailprice", "comment"]
    df = pd.read_csv("part.tbl", sep='|', names=col_names,
                     warn_bad_lines=True, error_bad_lines=False, index_col=False)
    df["name"].str.replace(regex_pattern, '').to_csv(
        "datasets/tpch/tpch_part_names.csv", index=False, header=False)

# This function will take a input file that contains strings one line at a time
# creates summary data structures for prefix, substring, suffix
# stores their selectivities
# and stores their triplets
# dataset_prefix is the name of the input file without .csv
# it will be used to generate outputs.
# eg. dblp_authors => dblp_authors.csv is the input file
# dblp_authors_prefix_count, dblp_authors_prefix_triplets contain the frequencies and triplets respectively


def prepare_dataset(folder_path, train_qs_prefix, total_prefix, file_name_prefix, triplet_name_prefix, threshold, max_l, prune_level):
    print("Processing data", train_qs_prefix)
    # functions = [summary_data_structures.get_all_prefixes, summary_data_structures.get_all_suffixes,
    #              summary_data_structures.get_all_substrings]
    # function_desc = ["prefix", "suffix", "substring"]
    functions = [summary_data_structures.get_all_prefixes]
    function_desc = ["prefix"]
    input_qs_file_name = folder_path + train_qs_prefix + ".csv"
    for index, fn in enumerate(functions):
        count_file_name = folder_path + total_prefix + "_" + function_desc[index] + "_counts.csv"
        count_file_name_w = folder_path + file_name_prefix + "_" + function_desc[index] + "_counts.csv"
        triplet_file_name = folder_path + triplet_name_prefix + "_" + function_desc[index] + "_triplets.csv"
        print("input  qs:", input_qs_file_name)
        print("total  qs:", count_file_name)
        print("save   qs:", count_file_name_w)
        print("save trpl:", triplet_file_name)
        assert os.path.exists(count_file_name), count_file_name
        if os.path.exists(count_file_name_w) and os.path.exists(triplet_file_name):
            print("Files already exist at",
                  count_file_name_w, "and", triplet_file_name)
            continue

        summary_data_structures.MAX_STR_SIZE = max_l
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
            if type(prune_level) == float:
                tree.prune(prune_level, seed=0)
            # summary_data_structures.store_selectivities(tree, count_file_name_w)
        else:
            summary_data_structures.store_selectivities(tree, count_file_name_w)
        # tree.print_tree()
        summary_data_structures.store_triplets(tree, triplet_file_name)
    print('\n\n')


# The following functions generate the raw files for 4 datasets.
# Note: these are already in the github repository
# load_and_save_dblp()
# load_imdb_movie_titles()
# load_imdb_movie_actors()
# load_and_save_tpch()

# The following functions generates the frequencies and triplets
# This function might take few minutes for large datasets :)
