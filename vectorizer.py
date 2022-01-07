import glob, os
import json

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


def read_data(path_X, path_Y):
    feature_dicts = []
    true_labels = []
    train_corpora = sorted(glob.iglob(path_X + 'train/' + '*'))
    test_corpora = sorted(glob.iglob(path_X + 'small-test/' + '*'))
    train_label_files = sorted(glob.iglob(path_Y + 'train/' + '*'))
    test_label_files = sorted(glob.iglob(path_Y + 'small-test/' + '*'))
    all_label_files = train_label_files + test_label_files
    n_train = 0
    for corpus in train_corpora:
        with open(corpus,'r') as f:
            fd = json.loads(f.read())
        for item in fd:
            feature_dicts.append(item)
            n_train += 1
    for corpus in test_corpora:
        with open(corpus,'r') as f:
            fd = json.loads(f.read())
        for item in fd:
            feature_dicts.append(item)
    for label_file in all_label_files:
        with open(label_file, 'r') as f:
            tls = f.readlines()
        for tl in tls:
            true_labels.append(tl)
    return feature_dicts, true_labels, n_train

def vectorize_data(word_feature_dicts, word_labels):
    vec = DictVectorizer()
    le = LabelEncoder()
    vectors = vec.fit_transform(word_feature_dicts)
    le.fit(word_labels)
    labels = le.transform(word_labels)
    return vectors, labels
