



# Author: Arthur Mensch

import timeit
import warnings
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
t0 = timeit.default_timer()

def read_data(path_X, path_Y):
    dev_list = ['ws212', 'ecpa']
    test_list = ['cb', 'ecpr', 'jhk', 'jhu', 'tgk', 'tgu', 'psk', 'psu', 'rondane',
                 'vm32', 'ws213', 'ws214', 'petet', 'wsj23']
    ignore_list = ['ntucle', 'omw', 'wlb03', 'wnb03']
    feature_dicts = []
    for corpus in glob.iglob(path_X + 'mrs'):
        with open(corpus,'r') as f:
            fd = json.loads(f.read())
        feature_dicts.append(fd)
    for corpus in glob.iglob(path_Y + 'mrs'):
        with open(corpus,'r') as f:
            labels = f.readlines()
    return feature_dicts, labels

def vectorize_data(word_feature_dicts):
    vectors = []
    vec = DictVectorizer()
    for fdl in word_feature_dicts:
        for fd in fdl:
            vectors.append(vec.fit_transform(fd))
    #fnames = vec.get_feature_names_out()
    return vectors

solver = "saga"

# Turn down for faster run time
n_samples = 5000

feature_dicts, true_labels = read_data('./output/contexts/', './output/true_labels/')
X = vectorize_data(feature_dicts)
le = LabelEncoder()
le.fit(true_labels)
y = le.transform(true_labels)

# X, y = fetch_20newsgroups_vectorized(subset="all", return_X_y=True)
# X = X[:n_samples]
# y = y[:n_samples]

# TODO Next: use the actual train-test split and all the data, see if this kind of code works out of the box...
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.1
)
train_samples, n_features = X_train.shape
n_classes = np.unique(y).shape[0]

print(
    "Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i"
    % (train_samples, n_features, n_classes)
)

models = {
    "ovr": {"name": "One versus Rest", "iters": [1, 2, 3]},
    "multinomial": {"name": "Multinomial", "iters": [1, 2, 5]},
}

for model in models:
    # Add initial chance-level values for plotting purpose
    accuracies = [1 / n_classes]
    times = [0]
    densities = [1]

    model_params = models[model]

    # Small number of epochs for fast runtime
    for this_max_iter in model_params["iters"]:
        print(
            "[model=%s, solver=%s] Number of epochs: %s"
            % (model_params["name"], solver, this_max_iter)
        )
        lr = LogisticRegression(
            solver=solver,
            multi_class=model,
            penalty="l1",
            max_iter=this_max_iter,
            random_state=42,
        )
        t1 = timeit.default_timer()
        lr.fit(X_train, y_train)
        train_time = timeit.default_timer() - t1

        y_pred = lr.predict(X_test)
        accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
        density = np.mean(lr.coef_ != 0, axis=1) * 100
        accuracies.append(accuracy)
        densities.append(density)
        times.append(train_time)
    models[model]["times"] = times
    models[model]["densities"] = densities
    models[model]["accuracies"] = accuracies
    print("Test accuracy for model %s: %.4f" % (model, accuracies[-1]))
    print(
        "%% non-zero coefficients for model %s, per class:\n %s"
        % (model, densities[-1])
    )
    print(
        "Run time (%i epochs) for model %s:%.2f"
        % (model_params["iters"][-1], model, times[-1])
    )

fig = plt.figure()
ax = fig.add_subplot(111)

for model in models:
    name = models[model]["name"]
    times = models[model]["times"]
    accuracies = models[model]["accuracies"]
    ax.plot(times, accuracies, marker="o", label="Model: %s" % name)
    ax.set_xlabel("Train time (s)")
    ax.set_ylabel("Test accuracy")
ax.legend()
fig.suptitle("Multinomial vs One-vs-Rest Logistic L1\nDataset %s" % "20newsgroups")
fig.tight_layout()
fig.subplots_adjust(top=0.85)
run_time = timeit.default_timer() - t0
print("Example run in %.3f s" % run_time)
plt.show()
