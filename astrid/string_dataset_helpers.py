import torch
import numpy as np
from torch.utils.data import Dataset


class StringDatasetHelper:
    def __init__(self):
        self.alphabets = ""
        self.alphabet_size = 0
        self.max_string_length = 0

    def extract_alphabets(self, file_name):
        f = open(file_name, "r")
        lines = f.read().splitlines()
        f.close()

        self.max_string_length = max(map(len, lines))
        # Make it to even for easier post processing and striding
        if self.max_string_length % 2 != 0:
            self.max_string_length += 1

        # Get the alphabets. Create a set with all strings, sort it and concatenate it
        self.alphabets = "".join(sorted(set().union(*lines)))
        self.alphabet_size = len(self.alphabets)

    # If alphabet is "abc" and for the word abba, this returns [0, 1, 1, 0]
    # 0 is the index for a and 1 is the index for b
    def string_to_ids(self, str_val):
        if len(str_val) > self.max_string_length:
            print("Warning: long string {} is passed. Subsetting to max length of {}".format(str_val, self.max_string_length))
            str_val = str[:self.max_string_length]
        indices = [self.alphabets.find(c) for c in str_val]
        if -1 in indices:
            raise ValueError("String {} contained unknown alphabets".format(str_val))
        return indices

    # Given a string (of any length), it outputs a fixed 2D tensor of size alphabet_size * max_string_length
    # If the string is shorter, the rest are filled with zeros
    # Each column corresponds to the i-th character of str_val
    # while each row corresponds to j-th character of self.alphabets
    # This encoding is good for CNN processing
    def string_to_tensor(self, str_val):
        string_indices = self.string_to_ids(str_val)
        one_hot_tensor = np.zeros((self.alphabet_size, self.max_string_length), dtype=np.float32)
        one_hot_tensor[np.array(string_indices), np.arange(len(string_indices))] = 1.0
        return torch.from_numpy(one_hot_tensor)


# Helper string to process file with three strings in each line
# It also automatically converts each string to a tensor
class TripletStringDataset(Dataset):
    def __init__(self, df, string_helper):
        # Convert data frame to numpy ndarray
        self.data = df.values
        self.string_helper = string_helper

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # Get the three strings, convert to tensor and return it
        #print("Item, data", item, self.data[item])
        # str(val) is a precaution for special cases such as strings like 'nan' which confuses pandas
        anchor, positive, negative = [self.string_helper.string_to_tensor(str(val)) for val in self.data[item]]
        return anchor, positive, negative

# Helper string to process file strings and automatically convert to embeddings


class StringSelectivityDataset(Dataset):
    def __init__(self, df, string_helper, embedding_model):
        self.strings = df["string"].values
        self.normalized_selectivities = df["normalized_selectivities"].values
        self.strings_as_tensors = []
        self.string_helper = string_helper
        self.embedding_model = embedding_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.eval()
        with torch.no_grad():
            for string in self.strings:
                string_as_tensor = string_helper.string_to_tensor(string).to(self.device)
                # By default embedding mode expects a tensor of [batch size x alphabet_size * max_string_length]
                # so create a "fake" dimension that converts the 2D matrix into a 3D tensor
                string_as_tensor = string_as_tensor.view(-1, *string_as_tensor.shape)
                self.strings_as_tensors.append(self.embedding_model(string_as_tensor).detach().cpu().numpy())
            self.strings_as_tensors = np.concatenate(self.strings_as_tensors)

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, item):
        return self.strings_as_tensors[item], self.normalized_selectivities[item]
