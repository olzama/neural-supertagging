
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

from vectorizer import read_data,vectorize_train_data

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")


def train_SVM(X, Y):
    clf = svm.LinearSVC() # Kernels would be too slow, so using liblinear SVM
    name = "svm-liblinear-l2-sq-hinge-1000"
    fit_serialize(X,Y,clf,name) # for models over 4GB, need to add protocol=4


def train_MaxEnt(X, Y):
    solver = "saga" # Another option is "sag"; it was also tried in development
    train_samples, n_features = X.shape
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
            # Small number of epochs for fast runtime; 100 is the default
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


def fit_serialize(X, Y, clf, name):
    t1 = timeit.default_timer()
    clf.fit(X, Y)
    train_time = timeit.default_timer() - t1
    print('Training time of {}: {}'.format(name, train_time))
    with open('models/' + name + '.model', 'wb') as f:
        pickle.dump(clf, f)


def test_model(model, X_test, Y_test, n_classes, corpus_id):
    with open(model, 'rb') as f:
        clf = pickle.load(f)
    y_pred = []
    Y_gold = []
    for sentence in X_test:
        pred = clf.predict(sentence)
        for l in pred:
            y_pred.append(l)
    for sentence in Y_test:
        for l in sentence:
            Y_gold.append(l)
    accuracy = np.sum(np.array(y_pred) == np.array(Y_gold)) / len(Y_gold)
    #density = np.mean(clf.coef_ != 0, axis=1) * 100
    print("Test accuracy for model %s: %.4f on corpus %s" % (model, accuracy, str(corpus_id)))
    return accuracy
    # print(
    #     "%% non-zero coefficients for model %s, per class:\n %s"
    #     % (model, densities[-1])
    # )



if __name__ == "__main__":
    feature_dicts, true_labels, n_train, test_sen_lengths, test_corpus_lengths\
        = read_data('./sample-data/contexts/', './sample-data/true_labels/')
    X, Y = vectorize_train_data(feature_dicts,true_labels)
    n_classes = np.unique(Y).shape[0]
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    test_corpora = []
    true_labels_per_corpus = []
    start = n_train
    corpus_start = 0
    for cl in test_corpus_lengths:
        X_test = []
        Y_test = []
        test_sentences = test_sen_lengths[corpus_start:corpus_start+cl]
        for l in test_sentences:
            X_test.append(X[start:start+l])
            Y_test.append(Y[start:start+l])
            start = start + l
        test_corpora.append(X_test)
        true_labels_per_corpus.append(Y_test)
        corpus_start = corpus_start + cl
    if sys.argv[1] == 'train':
        train_SVM(X_train,Y_train)
        train_MaxEnt(X_train, Y_train)
    elif sys.argv[1] == 'test':
        model_accuracies = []
        names = []
        times = []
        #densities = [1]
        for model in glob.iglob('models/' + '*'):
            accuracies = []
            t1 = timeit.default_timer()
            for i,tc in enumerate(test_corpora):
                acc = test_model(model,tc,true_labels_per_corpus[i],n_classes,i)
                accuracies.append(acc)
            test_time = timeit.default_timer() - t1
            print('Test time of {}: {}'.format(model, test_time))
            model_accuracies.append(accuracies)
            names.append(model)
            times.append(test_time)

        # ind = np.arange(len(names))
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.legend()
        # fig.suptitle("Accuracies on the ERG dev data")
        # ax.set_ylabel("Dev accuracy")
        # ax.set_xticks(ind, labels=names)
        # ax.plot(accuracies)
        # plt.savefig('accuracies.png')
