import glob
import json
import os.path
import sys
import pickle
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

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

def vectorize_nonautoreg(fp):
    vec = DictVectorizer()
    le = LabelEncoder()
    with open(fp,'rb') as f:
        table = pickle.load(f)
    X = []
    Y = []
    #X = table['ft']
    #Y = table['lt']
    for i, row in enumerate(table['ft']):
        X.extend(row)
        Y.extend(table['lt'][i])
    print("Number of observations in the vector matrix: {}".format(len(X)))
    vectors = vec.fit_transform(X)
    le.fit(Y)
    labels = le.transform(Y)
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    inv_le_dict = {v: k for k, v in le_dict.items()}
    return vectors,labels,vec,le_dict, inv_le_dict


if __name__ == "__main__":
    Path(sys.argv[1]+'/vectors').mkdir(parents=True, exist_ok=True)
    autoreg = sys.argv[2] == 'autoreg'
    data_file = sys.argv[3]
    if autoreg:
        X, Y, vectorizer, label_dict, inv_label_dict = vectorize_autoreg(sys.argv[1] + '/' + data_file)
    else:
        X, Y, vectorizer, label_dict, inv_label_dict = vectorize_nonautoreg(sys.argv[1]+ '/' + data_file)
    with open(sys.argv[1] + '/vectors/label-inv-dict', 'wb') as f:
        pickle.dump(inv_label_dict, f)
    with open(sys.argv[1]+'/vectors/vectorizer', 'wb') as f:
        pickle.dump(vectorizer,f)
        print("The saved vectorizer was created using sklearn version {}".format(vectorizer.__getstate__()['_sklearn_version']))
    with open(sys.argv[1]+'/vectors/label-dict','wb') as f:
        pickle.dump(label_dict,f)
    pickle_vectors(sys.argv[1]+'/vectors/', X, Y, data_file)



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
