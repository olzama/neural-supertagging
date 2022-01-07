import glob, os
import json

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


def read_data(path_X, path_Y):
    dev_list = ['ws212', 'ecpa']
    test_list = ['cb', 'ecpr', 'jhk', 'jhu', 'tgk', 'tgu', 'psk', 'psu', 'rondane',
                 'vm32', 'ws213', 'ws214', 'petet', 'wsj23']
    ignore_list = ['ntucle', 'omw', 'wlb03', 'wnb03']
    nontrain_list = dev_list + test_list + ignore_list
    X_train = []
    Y_train = []
    X_dev = []
    Y_dev = []
    X_test = []
    Y_test = []
    for corpus in sorted(glob.iglob(path_X + 'wsj01')):
        corpus_name = os.path.basename(corpus)
        with open(corpus,'r') as f:
            fd = json.loads(f.read())
        for item in fd:
            if corpus_name not in nontrain_list:
                X_train.append(item)
            elif corpus_name in dev_list:
                X_dev.append(item)
            elif corpus_name in test_list:
                X_test.append(item)
    for corpus in sorted(glob.iglob(path_Y + 'wsj01')):
        corpus_name = os.path.basename(corpus)
        with open(corpus, 'r') as f:
            tls = f.readlines()
        for tl in tls:
            if os.path.basename(corpus) not in nontrain_list:
                Y_train.append(tl)
            elif corpus_name in dev_list:
                Y_dev.append(tl)
            elif corpus_name in test_list:
                Y_test.append(tl)
    return X_train, Y_train, X_dev, Y_dev, X_test, Y_test

def vectorize_data(word_feature_dicts, word_labels):
    vec = DictVectorizer()
    le = LabelEncoder()
    vectors = vec.fit_transform(word_feature_dicts)
    le.fit(word_labels)
    labels = le.transform(word_labels)
    return vectors, labels


feature_dicts, true_labels = read_data('./output/contexts/', './output/true_labels/')
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = vectorize_data(feature_dicts, true_labels)
