
# https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_20newsgroups.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-20newsgroups-py
# Adapted from Arthur Mensch

import timeit
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import pickle,glob

import sys

import energyusage

from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

def train_SVM(X, Y,fp):
    clf = svm.LinearSVC() # Kernels would be too slow, so using liblinear SVM
    name = "svm-liblinear-l2-sq-hinge-1000"
    fit_serialize(X,Y,clf,name,fp) # for models over 4GB, need to add protocol=4

def train_MaxEnt(X, Y, fp, all=False):
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
                 "ovr": {"name": "OVR-L2-saga", "iters": [100]}}},
            #'elasticnet': {"multinomial": {"name": "Multinomial-ENet-saga", "iters": [100]}}},
            'sag': {'l2': {"multinomial": {"name": "Multinomial-L2-sag", "iters": [100]},
                  "ovr": {"name": "OVR-L2-sag", "iters": [100]}}}
        }
    else:
        models = {
            'saga': { 'l1': {"OVR": {"name": "OVR-L1", "iters": [100]}}}
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
                    fit_serialize(X, Y, clf, models[solver][penalty][model]["name"],fp)

def fit_serialize(X, Y, clf, name,fp):
    t1 = timeit.default_timer()
    clf.fit(X, Y)
    train_time = timeit.default_timer() - t1
    print('Training time of {}: {}'.format(name, train_time))
    with open(fp+'/' + name + '.model', 'wb') as f:
        pickle.dump(clf, f)
        print("The saved model was created using sklearn version {}".format(clf.__getstate__()['_sklearn_version']))


def test_autoreg(clf, name,vec,le_dict,table_path,inv_le_dict):
    all_preds_for_acc = []
    all_true_labels = []
    all_times = []
    errors = []
    for test_file in glob.iglob(table_path+'/**'):
        with open(test_file, 'rb') as f:
            table = pickle.load(f)
        times = []
        all_predictions = {}
        pred_list = []
        true_list = []
        for length in table:
            eprint('Processing sentences of length {}'.format(length))
            all_predictions[length] = np.empty_like(table[length]['lt'])
            for i, row in enumerate(table[length]['ft']):
                updated_row = update_row(list(row), all_predictions[length],i)
                x_i = vec.transform(updated_row)
                y_i = [ le_dict.get(lbl,-1) for lbl in table[length]['lt'][i] ]
                all_true_labels += y_i
                true_list += y_i
                t1 = timeit.default_timer()
                y_train_i = clf.predict(x_i)
                all_preds_for_acc += list(y_train_i)
                pred_list += list(y_train_i)
                test_time = timeit.default_timer() - t1
                for j,prediction in enumerate(y_train_i):
                    if prediction != y_i[j]:
                        errors.append((remove_tag_features(row[j]),inv_le_dict[prediction],table[length]['lt'][i][j]))
                times.append(test_time)
                all_predictions[length][i] = [inv_le_dict[pred] for pred in y_train_i]
        all_times.append(sum(times))
        acc = np.sum(np.array(true_list) == np.array(pred_list)) / len(pred_list)
        print('Accuracy of {} on {}: {}'.format(name, test_file, acc))
        eprint('Accuracy of {} on {}: {}'.format(name, test_file, acc))
    print('Total test time for {} on all datasets in {}: {}'.format(name, table_path, sum(all_times)))
    print('Accuracy of {} for all datasets in {}: {}'.format(name,table_path,
                                                                     np.sum(np.array(all_true_labels) ==
                                                                            np.array(all_preds_for_acc))
                                                             / len(all_true_labels)))
    eprint('Accuracy of {} for all datasets in {}: {}'.format(name,table_path,
                                                                     np.sum(np.array(all_true_labels) ==
                                                                            np.array(all_preds_for_acc))
                                                             / len(all_true_labels)))
    eprint('Total test time for {} on all datasets in {}: {}'.format(name, table_path, sum(all_times)))
    print('Number of predictions: {}'.format(len(all_preds_for_acc)))
    print('Number of true labels: {}'.format(len(all_true_labels)))
    print('Number of errors: {}'.format(len(errors)))
    classes = list(set([inv_le_dict[l] for l in all_preds_for_acc] + [inv_le_dict.get(l,'UNK') for l in all_true_labels]))
    cm = ConfusionMatrixDisplay.from_predictions(all_preds_for_acc, all_true_labels, display_labels=classes,
                                                 xticks_rotation="vertical")
    fig, ax = plt.subplots(figsize=(500, 500))
    cm.plot(ax=ax, xticks_rotation="vertical")
    plt.savefig(name + '-confmatrix.png')
    return errors

def remove_tag_features(obs):
    new_obs = {}
    for f in obs:
        if not f.startswith('tag-'):
            new_obs[f] = obs[f]
    return str(new_obs)

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

def test_model(clf,model,vec,le_dict,table_path,inv_le_dict):
    all_preds = []
    all_true_labels = []
    all_times = []
    errors = []
    for test_file in glob.iglob(table_path+'/**'):
        with open(test_file, 'rb') as f:
            table = pickle.load(f)
        X = vec.transform(table['ft'])
        y = [ le_dict.get(lbl,-1) for lbl in table['lt'] ]
        t1 = timeit.default_timer()
        y_pred = clf.predict(X)
        test_time = timeit.default_timer() - t1
        for j, prediction in enumerate(y_pred):
            if prediction != y[j]:
                errors.append((str(table['ft'][j]), inv_le_dict[prediction], inv_le_dict.get(y[j],'UNK')))
        all_times.append(test_time)
        all_preds += list(y_pred)
        all_true_labels += y
        acc = np.sum(np.array(y) == y_pred) / len(y)
        print('Accuracy of {} on {}: {}'.format(model, test_file, acc))
        eprint('Accuracy of {} on {}: {}'.format(model, test_file, acc))
    print('Total test time for {} on all datasets in {}: {}'.format(model, table_path, sum(all_times)))
    print('Accuracy of {} for all datasets in {}: {}'.format(model,table_path,
                                                                     np.sum(np.array(all_true_labels) ==
                                                                            np.array(all_preds))
                                                             / len(all_true_labels)))
    eprint('Accuracy of {} for all datasets in {}: {}'.format(model,table_path,
                                                                     np.sum(np.array(all_true_labels) ==
                                                                            np.array(all_preds))
                                                             / len(all_true_labels)))
    eprint('Total test time for {} on all datasets in {}: {}'.format(model, table_path, sum(all_times)))
    print('Number of predictions: {}'.format(len(all_preds)))
    print('Number of true labels: {}'.format(len(all_true_labels)))
    print('Number of errors: {}'.format(len(errors)))
    # classes = list(set([inv_le_dict[l] for l in all_preds] + [inv_le_dict.get(l,'UNK') for l in all_true_labels]))
    # cm = ConfusionMatrixDisplay.from_predictions(all_preds, all_true_labels, display_labels=classes,
    #                                              xticks_rotation="vertical")
    # fig, ax = plt.subplots(figsize=(500, 500))
    # cm.plot(ax=ax, xticks_rotation="vertical")
    # plt.savefig(model + '-confmatrix.png')
    return errors



def load_vectors(path_to_vecs, path_to_labels):
    with open(path_to_vecs, 'rb') as vf:
        vecs = pickle.load(vf)
    with open(path_to_labels,'rb') as lf:
        labels = pickle.load(lf)
    return vecs, labels

if __name__ == "__main__":
    if sys.argv[1] == 'train':
        Path(sys.argv[2] + '/models').mkdir(parents=True, exist_ok=True)
        X, Y = load_vectors(sys.argv[2]+'/vectors/X_train', sys.argv[2]+'/vectors/Y_train')
        with open(sys.argv[2] +'/vectors/vectorizer','rb') as f:
            vec = pickle.load(f)
        # with open('/Users/olzama/Desktop/cur-features.txt', 'w') as f:
        #     for feat in vec.feature_names_:
        #         f.write(feat + '\n')
        #energyusage.evaluate(train_SVM,X,Y,sys.argv[2] + '/models')
        train_SVM(X,Y,sys.argv[2] + '/models')
        #train_MaxEnt(X,Y,sys.argv[2] + '/models',all=True)
    elif sys.argv[1] == 'test':
        to_test = sys.argv[2] + '/labeled-data/' + sys.argv[3]
        with open(sys.argv[2] +'/vectors/vectorizer','rb') as f:
            vec = pickle.load(f)
            print("The loaded vectorizer was created using sklearn version {}".format(
                vec.__getstate__()['_sklearn_version']))
            print('Number of features: {}'.format(len(vec.feature_names_)))
        with open(sys.argv[2]+'/vectors/label-dict', 'rb') as f:
            le_dict = pickle.load(f)
            print('Number of classes: {}'.format(len(le_dict)))
        with open(sys.argv[2] + '/vectors/label-inv-dict', 'rb') as f:
            inv_le_dict = pickle.load(f)
            print('Number of classes in the inverse dictionary: {}'.format(len(inv_le_dict)))
        models = glob.iglob(sys.argv[2] + '/models/*.model')
        for model in models:
            with open(model,'rb') as f:
                clf = pickle.load(f)
                print("The loaded model was created using sklearn version {}".format(clf.__getstate__()['_sklearn_version']))
            autoreg = sys.argv[4] == 'autoreg'
            if autoreg:
                errors = test_autoreg(clf,model,vec,le_dict,to_test,inv_le_dict)
            else:
                errors = test_model(clf,model,vec,le_dict,to_test,inv_le_dict)
            with open(model + '-errors.txt', 'w') as f:
                for e in sorted(errors):
                    e_str = 'Observation: {}, Prediction: {}, True label: {}'.format(e[0], e[1], e[2])
                    f.write(e_str + '\n')

