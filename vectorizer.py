import glob
import json

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


def read_data(path_X, path_Y):
    feature_dicts = []
    true_labels = []
    test_sen_lengths = []
    test_corpus_lengths = {}
    train_corpora = sorted(glob.iglob(path_X + 'train/' + '*'))
    test_corpora = sorted(glob.iglob(path_X + 'dev/' + '*'))
    train_label_files = sorted(glob.iglob(path_Y + 'train/' + '*'))
    test_label_files = sorted(glob.iglob(path_Y + 'dev/' + '*'))
    all_label_files = train_label_files + test_label_files
    n_train = 0
    for corpus in train_corpora:
        with open(corpus,'r') as f:
            fd = json.loads(f.read())
        for sentence in fd:
            if sentence[0] != 'NO PARSE':
                for item in sentence:
                    feature_dicts.append(item)
                    n_train += 1
    for corpus in test_corpora:
        with open(corpus,'r') as f:
            fd = json.loads(f.read())
        test_corpus_lengths[corpus] = len(fd)
        for sentence in fd:
            if sentence[0] != 'NO PARSE':
                test_sen_lengths.append(len(sentence))
                for item in sentence:
                    feature_dicts.append(item)
    for label_file in all_label_files:
        with open(label_file, 'r') as f:
            tls = f.readlines()
        for tl in tls:
            if tl != '\n':
                true_labels.append(tl)
    return feature_dicts, true_labels, n_train, test_sen_lengths, test_corpus_lengths

'''
This function needs to treat the entire data set as a matrix,
so training and test data must have the same dimensions.
'''
def vectorize_data(word_feature_dicts, word_labels, n_train):
    vec = DictVectorizer()
    le = LabelEncoder()
    train_vectors = vec.fit_transform(word_feature_dicts[:n_train])
    test_vectors = vec.transform(word_feature_dicts[n_train:])
    le.fit(word_labels)
    labels = le.transform(word_labels)
    return train_vectors, test_vectors, labels

def vectorize_test_data(word_feature_dicts, word_labels):
    pass

# Below just a small example I used to inspect the features.

# pos_window = [     {
#          'word-2': 'the',
#          'pos-2': 'DT',
#          'word-1': 'cat',
#          'pos-1': 'NN',
#          'word+1': 'on',
#          'pos+1': 'PP',
#      }, {
#          'word-2': 'the',
#          'pos-2': 'DT',
#          'word-1': 'dog',
#          'pos-1': 'NN',
#          'word+1': 'on',
#          'pos+1': 'PP',
#      },
#
#  ]

# vec = DictVectorizer()
# pos_vectorized = vec.fit_transform(pos_window)
# arr = pos_vectorized.toarray()
# names = vec.get_feature_names_out()
