import string
import nltk
import numpy as np
from sklearn.utils import shuffle

def init_weights(fi, fo):
    return np.random.randn(fi, fo) / np.sqrt(fi + fo)


def remove_punctuation(s):
    return s.translate(string.punctuation)


def get_poetry_classifier_data(samples_per_class=700):
    rf_data = open('robert_frost.txt')
    ea_data = open('edgar_allan.txt')

    X = []
    pos2idx = {}
    cur_idx = 0

    # Remove punctuation from both data
    rf_text = [remove_punctuation(s.strip().lower()) for s in rf_data]
    ea_text = [remove_punctuation(s.strip().lower()) for s in ea_data]

    # Loop through to form sequences of pos_tag for both the data
    rf_line_count = 0
    for s in rf_text:
        tokens = nltk.pos_tag(s.split())
        if tokens:
            seq = []
            for (label, val) in tokens:
                if val not in pos2idx:
                    pos2idx[val] = cur_idx
                    cur_idx += 1
                seq += [pos2idx[val]]
            X.append(seq)
            rf_line_count += 1
            if rf_line_count == samples_per_class:
                break

    ea_line_count = 0
    for s in ea_text:
        tokens = nltk.pos_tag(s.split())
        if tokens:
            seq = []
            for (label, val) in tokens:
                if val not in pos2idx:
                    pos2idx[val] = cur_idx
                    cur_idx += 1
                seq += [pos2idx[val]]
            X.append(seq)
            ea_line_count += 1
            if ea_line_count == samples_per_class:
                break

    # Set Y to 0 for robert frost poems and 1 for edgar allan poems
    Y = np.array([0] * rf_line_count + [1] * ea_line_count).astype(np.int32)
    X, Y = shuffle(X, Y)
    return X, Y, len(pos2idx)
