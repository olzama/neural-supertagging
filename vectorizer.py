import glob
import json
import os.path
import sys
import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

import ERG_Corpus

def read_data(path_X, path_Y):
    feature_dicts = []
    true_labels = []
    sen_lengths = []
    corpora = []
    label_files = []
    if os.path.isdir(path_X) and os.path.isdir(path_Y):
        corpora = sorted(glob.iglob(path_X + '/*'))
        label_files = sorted(glob.iglob(path_Y + '/*'))
    elif os.path.isfile(path_X) and os.path.isfile(path_Y):
        corpora=glob.glob(path_X)
        label_files=glob.glob(path_Y)
    for corpus in corpora:
        with open(corpus,'r') as f:
            fd = json.loads(f.read())
        for sentence in fd:
            sen_lengths.append(len(sentence))
            if sentence[0] != 'NO PARSE':
                for item in sentence:
                    feature_dicts.append(item)
    for label_file in label_files:
        with open(label_file, 'r') as f:
            tls = f.readlines()
        for tl in tls:
            if tl != '\n':
                true_labels.append(tl)
    return feature_dicts, true_labels, sen_lengths

'''
This function needs to treat the entire data set as a matrix,
so training and test data must have the same dimensions.
'''
def vectorize_train_data(word_feature_dicts, word_labels):
    vec = DictVectorizer()
    le = LabelEncoder()
    vectors = vec.fit_transform(word_feature_dicts)
    le.fit(word_labels)
    labels = le.transform(word_labels)
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    return vectors, labels, vec, le_dict

def vectorize_test_data(word_feature_dicts, word_labels, vec, le_dict):
    vectors = vec.transform(word_feature_dicts)
    labels = []
    unknowns = 0
    for l in word_labels:
        tl = le_dict.get(l,-1)
        if tl == -1:
            unknowns += 1
        labels.append(tl) # Return -1 is unknown value
    return vectors, labels, unknowns

def pickle_vectors(path,X, Y, suf):
    with open(path + 'X_'+suf, 'wb') as xf:
        pickle.dump(X, xf)
    with open(path + 'Y_'+suf, 'wb') as yf:
        pickle.dump(Y, yf)

def vectorize_autoreg(fp):
    vec = DictVectorizer()
    le = LabelEncoder()
    X = []
    Y = []
    with open(fp,'rb') as f:
        table = pickle.load(f)
    for len in table:
        for i,row in enumerate(table[len]['ft']):
            X += list(row)
            Y += list(table[len]['lt'][i])
    vectors = vec.fit_transform(X)
    le.fit(Y)
    labels = le.transform(Y)
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    inv_le_dict = {v: k for k, v in le_dict.items()}
    return vectors,labels,vec,le_dict, inv_le_dict

if __name__ == "__main__":
    # See sample data for the expected format.
    Path(sys.argv[1]+'/vectors').mkdir(parents=True, exist_ok=False)
    X, Y, vectorizer, label_dict, inv_label_dict = vectorize_autoreg(sys.argv[1]+'/labeled-data/train/train')
    with open(sys.argv[1] + '/vectors/label-inv-dict', 'wb') as f:
        pickle.dump(inv_label_dict, f)
    with open(sys.argv[1]+'/vectors/vectorizer', 'wb') as f:
        pickle.dump(vectorizer,f)
    with open(sys.argv[1]+'/vectors/label-dict','wb') as f:
        pickle.dump(label_dict,f)
    pickle_vectors(sys.argv[1]+'/vectors/', X, Y, 'train')



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
