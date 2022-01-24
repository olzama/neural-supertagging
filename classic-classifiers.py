
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py
# Adapted from Arthur Mensch

import timeit
import warnings

import numpy as np

import pickle,glob

import sys

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn import svm

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


def train_SVM(X, Y):
    clf = svm.LinearSVC() # Kernels would be too slow, so using liblinear SVM
    name = "svm-liblinear-l2-sq-hinge-1000"
    fit_serialize(X,Y,clf,name) # for models over 4GB, need to add protocol=4


def train_MaxEnt(X, Y):
    solver = "saga" # Another option is "sag"; it was also tried in development
    train_samples, n_features = X.shape
    n_classes = np.unique(Y).shape[0]
    print(
        "Dataset ERG treebanks, train_samples=%i, n_features=%i, n_classes=%i"
        % (train_samples, n_features, n_classes)
    )

    # All MaxEnt models tried in development:
    # models = {
    #     'l1': {"multinomial": {"name": "Multinomial-L1", "iters": [10]},
    #            "ovr": {"name": "One versus Rest-L1", "iters": [10]}},
    #     'l2': {"multinomial": {"name": "Multinomial-L2", "iters": [10]},
    #            "ovr": {"name": "One versus Rest-L2", "iters": [10]}},
    #     'elasticnet': {"multinomial": {"name": "Multinomial-ENet", "iters": [10]}},
    # }

    # The overall best MaxEnt model (high accuracy, low train time, best test time, out of other MaxEnts
    # This assumes SAGA solver; with SAG OVR L1, can get higher accuracy but training time is huge.
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
                    #l1_ratio=0.5 # only for elastic-net
                )
                model_name = models[penalty][model]["name"] + '-' + solver
                fit_serialize(X, Y, clf, model_name)


def fit_serialize(X, Y, clf, name):
    t1 = timeit.default_timer()
    clf.fit(X, Y)
    train_time = timeit.default_timer() - t1
    print('Training time of {}: {}'.format(name, train_time))
    with open('models/' + name + '.model', 'wb') as f:
        pickle.dump(clf, f)



def test_model(clf, X_test, Y_test, num_sentences):
    y_pred = []
    Y_gold = []
    times = []
    for i,sentence in enumerate(X_test):
        t1 = timeit.default_timer()
        pred = clf.predict(sentence)
        test_time = timeit.default_timer() - t1
        times.append(test_time)
        y_pred = y_pred + list(pred)
    t_sum = sum(times)
    avg_time = t_sum/num_sentences
    print('Test time of {}: {}; {} average per sentence'.format(model, t_sum, avg_time))
    for sentence_labels in Y_test:
        Y_gold = Y_gold + list(sentence_labels)
    accuracy = np.sum(np.array(y_pred) == np.array(Y_gold)) / len(Y_gold)
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
        #train_SVM(X_train,Y_train)
        train_MaxEnt(X, Y)
    elif sys.argv[1] == 'test':
        with open(sys.argv[2], 'rb') as cf:
            corpus = pickle.load(cf)
        model_accuracies = []
        for model in glob.iglob('models/' + '*'):
            with open(model, 'rb') as f:
                clf = pickle.load(f)
            accuracies = []
            acc = test_model(clf,corpus.X,corpus.Y,len(corpus.sen_lengths))
            accuracies.append(acc)
            model_accuracies.append(accuracies)
