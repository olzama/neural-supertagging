import glob
import json
import sys
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

import ERG_Corpus

def read_data(path_X, path_Y):
    feature_dicts = []
    true_labels = []
    sen_lengths = []
    corpora = sorted(glob.iglob(path_X + '*'))
    label_files = sorted(glob.iglob(path_Y + '*'))
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
    return vectors, labels, vec, le

def vectorize_test_data(word_feature_dicts, word_labels, vec, le):
    vectors = vec.transform(word_feature_dicts)
    labels = le.transform(word_labels)
    return vectors, labels


def pickle_vectors(path,X, Y, suf):
    with open(path + 'X_'+suf, 'wb') as xf:
        pickle.dump(X, xf)
    with open(path + 'Y_'+suf, 'wb') as yf:
        pickle.dump(Y, yf)


if __name__ == "__main__":
    # See sample data for the expected format.
    # The paths must end with '/' (be directories)
    feature_dicts, true_labels, sen_lengths = read_data(sys.argv[1], sys.argv[2])
    if sys.argv[3]=='train':
        X, Y, vectorizer, label_encoder = vectorize_train_data(feature_dicts,true_labels)
        with open(sys.argv[4]+'/vectorizer', 'wb') as f:
            pickle.dump(vectorizer,f)
        with open(sys.argv[4]+'/label-encoder','wb') as f:
            pickle.dump(label_encoder,f)
        pickle_vectors(sys.argv[4], X, Y, 'train')
    if sys.argv[3]=='test':
        with open(sys.argv[4]+'/vectorizer', 'rb') as f:
            vec = pickle.load(f)
        with open(sys.argv[4]+'/label-encoder', 'rb') as f:
            le = pickle.load(f)
        X,Y = vectorize_test_data(feature_dicts,true_labels, vec, le)
        c = ERG_Corpus.ERG_Corpus(sys.argv[5],X,Y,sen_lengths)
        with open(sys.argv[4]+'/'+c.name, 'wb') as cf:
            pickle.dump(c,cf)



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
