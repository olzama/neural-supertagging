
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py
# Adapted from Arthur Mensch

import timeit
import warnings

import numpy as np

import pickle,glob

import sys,os

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

def train_SVM(X, Y):
    clf = svm.LinearSVC() # Kernels would be too slow, so using liblinear SVM
    name = "svm-liblinear-l2-sq-hinge-1000"
    fit_serialize(X,Y,clf,name) # for models over 4GB, need to add protocol=4

def train_MaxEnt(X, Y, all=False):
    train_samples, n_features = X.shape
    n_classes = np.unique(Y).shape[0]
    print(
        "Dataset ERG treebanks, train_samples=%i, n_features=%i, n_classes=%i"
        % (train_samples, n_features, n_classes)
    )

    if all:
        # All MaxEnt models tried in development:
        models = { 'saga': {
            'l1': {"multinomial": {"name": "Multinomial-L1-saga", "iters": [100]},
                  "ovr": {"name": "OVR-L1-saga", "iters": [100]}},
            'l2': {"multinomial": {"name": "Multinomial-L2-saga", "iters": [100]},
                 "ovr": {"name": "OVR-L2-saga", "iters": [100]}},
            'elasticnet': {"multinomial": {"name": "Multinomial-ENet-saga", "iters": [100]}}},
            'sag': {'l2': {"multinomial": {"name": "Multinomial-L2-sag", "iters": [100]},
                  "ovr": {"name": "OVR-L2-sag", "iters": [100]}}}
        }
    else:
        models = {
            'saga': { 'l2': {"multinomial": {"name": "Multinomial-L2", "iters": [1]}}}
        }

    for solver in models:
        for penalty in models[solver]:
            for model in models[solver][penalty]:
                model_params = models[solver][penalty][model]
                for this_max_iter in model_params["iters"]:
                    print(
                        "[model=%s, solver=%s] Number of epochs: %s"
                        % (model_params["name"], solver, this_max_iter)
                    )
                    clf = LogisticRegression(
                        solver=solver,
                        multi_class=model,
                        penalty=penalty,
                        max_iter=this_max_iter,
                        random_state=42,
                        l1_ratio=0.5 # only for elastic-net
                    )
                    fit_serialize(X, Y, clf, models[solver][penalty][model]["name"])

def fit_serialize(X, Y, clf, name):
    t1 = timeit.default_timer()
    clf.fit(X, Y)
    train_time = timeit.default_timer() - t1
    print('Training time of {}: {}'.format(name, train_time))
    with open('models-dridan/' + name + '.model', 'wb') as f:
        pickle.dump(clf, f)


def test_autoreg(clf, name,vec,le_dict,table_path,le_inv_dict):
    with open(table_path, 'rb') as f:
        table = pickle.load(f)
    times = []
    accuracies = []
    all_predictions = {}
    for length in table:
        eprint('Processing sentences of length {}'.format(length))
        all_predictions[length] = np.empty_like(table[length]['lt'])
        for i, row in enumerate(table[length]['ft']):
            updated_row = update_row(list(row), all_predictions[length],i)
            x_i = vec.transform(updated_row)
            y_i = [ le_dict.get(lbl,-1) for lbl in table[length]['lt'][i] ]
            t1 = timeit.default_timer()
            y_train_i = clf.predict(x_i)
            test_time = timeit.default_timer() - t1
            times.append(test_time)
            train_acc_i = np.sum(y_i == np.array(y_train_i)) / len(y_i)
            #print('Processed row {}; accuracy {}'.format(i,train_acc_i))
            accuracies.append(train_acc_i)
            all_predictions[length][i] = [inv_le_dict[pred] for pred in y_train_i]
    print('Test time of {}: {}'.format(name, sum(times)))
    print('Average accuracy of {} for all tokens in {}: {}'.format(name,table_path,sum(accuracies)/len(accuracies)))
    eprint('Test time of {}: {}'.format(name, sum(times)))
    eprint('Average accuracy of {} for all tokens in {}: {}'.format(name,table_path,sum(accuracies)/len(accuracies)))

def update_row(row,ys,i):
    new_row = []
    if i > 0:
        for j,obs in enumerate(row):
            new_obs = {}
            for f in obs:
                if f == 'tag-1':
                    new_obs[f] = ys[i-1][j]
                elif f == 'tag-2' and i > 1:
                    new_obs[f] = ys[i-2][j]
                else:
                    new_obs[f] = obs[f]
            new_row.append(new_obs)
        return new_row
    else:
        return row

def test_model(clf, X_test, Y_test, num_sentences):
    t1 = timeit.default_timer()
    y_pred = clf.predict(X_test)
    test_time = timeit.default_timer() - t1
    avg_time = test_time/num_sentences
    print('Test time of {}: {}; {} average per sentence'.format(model, test_time, avg_time))
    accuracy = np.sum(np.array(y_pred) == np.array(Y_test)) / len(Y_test)
    print("Test accuracy for model %s: %.4f" % (model, accuracy))
    return accuracy

def load_vectors(path_to_vecs, path_to_labels):
    with open(path_to_vecs, 'rb') as vf:
        vecs = pickle.load(vf)
    with open(path_to_labels,'rb') as lf:
        labels = pickle.load(lf)
    return vecs, labels

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        X, Y = load_vectors(sys.argv[2], sys.argv[3])
        #train_SVM(X,Y)
        train_MaxEnt(X,Y,all=True)
    elif sys.argv[1] == 'test':
        autoregressive = sys.argv[5] == 'autoreg'
        corpora = []
        if os.path.isdir(sys.argv[2]):
            corpora = sorted(glob.iglob(sys.argv[2] + '/*'))
        elif os.path.isfile(sys.argv[2]):
            corpora = glob.glob(sys.argv[2])
        if not autoregressive:
            for c in corpora:
                    with open(c, 'rb') as cf:
                        corpus = pickle.load(cf)
                    print('Testing corpus {} which has {} unknown labels'.format(corpus.name, corpus.unk))
                    for model in glob.iglob('models/' + '*'):
                        with open(model, 'rb') as f:
                            clf = pickle.load(f)
                        acc = test_model(clf,corpus.X,corpus.Y,len(corpus.sen_lengths))
        else:
            with open(sys.argv[3]+'/vectorizer','rb') as f:
                vec = pickle.load(f)
            with open(sys.argv[3]+'/label-dict', 'rb') as f:
                le_dict = pickle.load(f)
            with open(sys.argv[3] + '/label-inv-dict', 'rb') as f:
                inv_le_dict = pickle.load(f)
            models = glob.iglob('models-dridan/*')
            for model in models:
                with open(model,'rb') as f:
                    clf = pickle.load(f)
                test_autoreg(clf,model,vec,le_dict,sys.argv[4],inv_le_dict)
