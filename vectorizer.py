import glob, os
import json

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder


def read_data(path_X, path_Y):
    dev_list = ['ws212', 'ecpa']
    test_list = ['cb', 'ecpr', 'jhk', 'jhu', 'tgk', 'tgu', 'psk', 'psu', 'rondane',
                 'vm32', 'ws213', 'ws214', 'petet', 'wsj23']
    ignore_list = ['ntucle', 'omw', 'wlb03', 'wnb03']
    skip_list = dev_list + test_list + ignore_list
    feature_dicts = []
    labels = []
    for corpus in sorted(glob.iglob(path_X + 'wsj01')):
        if os.path.basename(corpus) not in skip_list:
            with open(corpus,'r') as f:
                fd = json.loads(f.read())
            for item in fd:
                feature_dicts.append(item)
    for corpus in sorted(glob.iglob(path_Y + 'wsj01')):
        if os.path.basename(corpus) not in skip_list:
            with open(corpus,'r') as f:
                tls = f.readlines()
            for tl in tls:
                labels.append(tl)
    return feature_dicts, labels

def vectorize_data(word_feature_dicts, word_labels):
    vec = DictVectorizer()
    le = LabelEncoder()
    vectors = vec.fit_transform(word_feature_dicts)
    le.fit(word_labels)
    labels = le.transform(word_labels)
    return vectors, labels


feature_dicts, true_labels = read_data('./output/contexts/', './output/true_labels/')
X,y = vectorize_data(feature_dicts, true_labels)
