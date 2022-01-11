
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py
# Adapted from Arthur Mensch

import timeit
import warnings

import matplotlib.pyplot as plt
import numpy as np

import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn import svm

from vectorizer import read_data,vectorize_data

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
t0 = timeit.default_timer()


def train_SVM(X_train, Y_train):
    svm_clf = svm.SVC(kernel="linear", C=1.0)
    svm_clf.fit(X_train, Y_train)
    with open('svm.model', 'wb') as f:
        pickle.dump(svm_clf, f)
    #svm_pred = svm_clf.predict(X_test)
    #accuracy = np.sum(svm_pred == Y_test) / Y_test.shape[0]
    #print("Test accuracy for SVM: %.4f" % (accuracy))

def train_MaxEnt_SAGA(X_train, Y_train):
    solver = "saga"

    train_samples, n_features = X_train.shape


    print(
        "Dataset ERG treebanks, train_samples=%i, n_features=%i, n_classes=%i"
        % (train_samples, n_features, n_classes)
    )

    # models = {
    #     'l1': {"multinomial": {"name": "Multinomial-L1", "iters": [10]},
    #            "ovr": {"name": "One versus Rest-L1", "iters": [10]}},
    #     'l2': {"multinomial": {"name": "Multinomial-L2", "iters": [10]},
    #            "ovr": {"name": "One versus Rest-L2", "iters": [10]}},
    #     'elasticnet': {"multinomial": {"name": "Multinomial-ENet", "iters": [10]}},
    # }

    models = {
        'elasticnet': {"multinomial": {"name": "Multinomial-ENet", "iters": [10]}},
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
                lr = LogisticRegression(
                    solver=solver,
                    multi_class=model,
                    penalty=penalty,
                    max_iter=this_max_iter,
                    random_state=42,
                    l1_ratio=0.5
                )
                t1 = timeit.default_timer()
                lr.fit(X_train, Y_train)
                train_time = timeit.default_timer() - t1
                print('Training time of {}: {}'.format(models[penalty][model]["name"],train_time))
                with open('models/'+models[penalty][model]["name"]+'.model','wb') as f:
                    pickle.dump(lr,f)


def test_model(model, X_test, Y_test, n_classes):

    with open(model, 'rb') as f:
        clf = pickle.load(f)

    y_pred = clf.predict(X_test)
    accuracy = np.sum(y_pred == Y_test) / Y_test.shape[0]
    #density = np.mean(clf.coef_ != 0, axis=1) * 100
    print("Test accuracy for model %s: %.4f" % (model, accuracy))
    # print(
    #     "%% non-zero coefficients for model %s, per class:\n %s"
    #     % (model, densities[-1])
    # )



if __name__ == "__main__":
    feature_dicts, true_labels, n_train = read_data('./data/contexts/', './data/true_labels/')
    X, Y = vectorize_data(feature_dicts,true_labels)
    n_classes = np.unique(Y).shape[0]
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_test = X[n_train:]
    Y_test = Y[n_train:]


    train_MaxEnt_SAGA(X_train,Y_train)

    #test_model('models/Multinomial-L1.model',X_test,Y_test,n_classes)


    # Add initial chance-level values for plotting purpose
    #accuracies = [1 / n_classes]
    #times = [0]
    #densities = [1]

    # accuracies.append(accuracy)
    # densities.append(density)
    #
    # times.append(train_time)
    # models[penalty][model]["times"] = times
    # models[penalty][model]["densities"] = densities
    # models[penalty][model]["accuracies"] = accuracies

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    #
    # for penalty in models:
    #     for model in models[penalty]:
    #         name = models[penalty][model]["name"]
    #         times = models[penalty][model]["times"]
    #         accuracies = models[penalty][model]["accuracies"]
    #         ax.plot(times, accuracies, marker="o", label="Model: %s" % name)
    #         ax.set_xlabel("Train time (s)")
    #         ax.set_ylabel("Test accuracy")
    # ax.legend()
    # fig.suptitle("Multinomial Logistic L1-L2\nDataset %s" % "WS212")
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.85)
    # run_time = timeit.default_timer() - t0
    # print("Example run in %.3f s" % run_time)
    # plt.savefig('maxent.png')
