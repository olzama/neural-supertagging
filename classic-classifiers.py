
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


warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

def train_SVM(X, Y):
    clf = svm.LinearSVC() # Kernels would be too slow, so using liblinear SVM
    name = "svm-liblinear-l2-sq-hinge-1000"
    fit_serialize(X,Y,clf,name) # for models over 4GB, need to add protocol=4

def train_MaxEnt(X, Y, all=False):
    solver = "saga" # Another option is "sag"; it was also tried in development
    # train_samples, n_features = X.shape
    # n_classes = np.unique(Y).shape[0]
    # print(
    #     "Dataset ERG treebanks, train_samples=%i, n_features=%i, n_classes=%i"
    #     % (train_samples, n_features, n_classes)
    # )

    if all:
        # All MaxEnt models tried in development:
        models = {
            'l1': {"multinomial": {"name": "Multinomial-L1", "iters": [100]},
                  "ovr": {"name": "One versus Rest-L1", "iters": [100]}},
            'l2': {"multinomial": {"name": "Multinomial-L2", "iters": [100]},
                  "ovr": {"name": "One versus Rest-L2", "iters": [100]}}#,
            #'elasticnet': {"multinomial": {"name": "Multinomial-ENet", "iters": [100]}},
        }
    else:
        models = {
            'l2': {"multinomial": {"name": "Multinomial-L2", "iters": [1]}}
        }

    for penalty in models:
        for model in models[penalty]:
            model_params = models[penalty][model]
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
                model_name = models[penalty][model]["name"] + '-' + solver
                fit_serialize(X, Y, clf, model_name)

def train_autoreg_MaxEnt():
    solver = "saga"
    models = {
            'l2': {"multinomial": {"name": "Multinomial-L2", "iters": [1]}}
    }

    for penalty in models:
        for model in models[penalty]:
            model_params = models[penalty][model]
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
                model_name = models[penalty][model]["name"] + '-' + solver
                vec = DictVectorizer()
                le = LabelEncoder()
                fit_serialize_autoreg(clf,model_name,vec,le)


def fit_serialize(X, Y, clf, name):
    t1 = timeit.default_timer()
    clf.fit(X, Y)
    train_time = timeit.default_timer() - t1
    print('Training time of {}: {}'.format(name, train_time))
    with open('models/' + name + '.model', 'wb') as f:
        pickle.dump(clf, f)

def vectorize_autoreg(fp):
    vec = DictVectorizer()
    le = LabelEncoder()
    all_obs = []
    all_labels = []
    with open('./output/lextypes', 'rb') as f:
        lextypes = pickle.load(f)
    with open(fp + 'tables_by_length', 'rb') as f:
        table = pickle.load(f)
    for length in table:
        for i, row in enumerate(table[length]['ft']):
            all_obs.append(row)
            le.fit(table[length]['lt'][i])
            le.transform(table[length]['lt'][i])
    for lt in list(lextypes):
        vec.feature_names_.update({'tag-1':lt,'tag-2':lt})
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    le_inv_dict = {v: k for k, v in le_dict.items()}
    X = []
    ys = []
    # for j, length in enumerate(table):
    #     lt = table[j]
    #     for row in ft:
    #         # for obs in row:
    #         #     print(obs)
    #         #     print(vec.transform(obs))
    #         X.append(vec.transform(row))
    #     for i,labels in enumerate(lt):
    #         ys.append([])
    #         for lbl in labels:
    #             if lbl:
    #                 ys[i].append(le_dict[lbl])
    #             else:
    #                 ys[i].append(None)
    return X, ys, vec, le_dict, le_inv_dict

def fit_serialize_autoreg(clf, name,vec,lbl_enc):
    t1 = timeit.default_timer()
    all_obs = []
    all_labels = []
    with open('./output/lextypes', 'rb') as f:
        lextypes = pickle.load(f)
    with open('./output/by-length/tables_by_length', 'rb') as f:
        table = pickle.load(f)
    all_obs = []
    all_predictions = []
    for length in table:
        for i, row in enumerate(table[length]['ft']):
            updated_row = update_row(list(row), all_predictions,i)
            all_obs += updated_row
            vec.fit_transform(all_obs)
            x_i = vec.transform(updated_row)
            y_i = table[length]['lt'][i]
            clf.fit(x_i, y_i)
            y_train_i = clf.predict(x_i)
            train_acc_i = np.sum(y_i == np.array(y_train_i)) / len(y_i)
            print('Processed row {}; accuracy {}'.format(i,train_acc_i))
            all_predictions.append(y_train_i)
    train_time = timeit.default_timer() - t1
    print('Training time of {}: {}'.format(name, train_time))
    with open('models/' + name + '.model', 'wb') as f:
        pickle.dump(clf, f)

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
        autoregressive = sys.argv[4] == 'autoreg'
        #X, Y = load_vectors(sys.argv[2], sys.argv[3])
        #train_SVM(X,Y)
        train_autoreg_MaxEnt()
    elif sys.argv[1] == 'test':
        corpora = []
        if os.path.isdir(sys.argv[2]):
            corpora = sorted(glob.iglob(sys.argv[2] + '/*'))
        elif os.path.isfile(sys.argv[2]):
            corpora = glob.glob(sys.argv[2])
        for c in corpora:
            with open(c, 'rb') as cf:
                corpus = pickle.load(cf)
            print('Testing corpus {} which has {} unknown labels'.format(corpus.name, corpus.unk))
            for model in glob.iglob('models/' + '*'):
                with open(model, 'rb') as f:
                    clf = pickle.load(f)
                acc = test_model(clf,corpus.X,corpus.Y,len(corpus.sen_lengths))
