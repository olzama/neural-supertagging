import sys
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import itertools


def systematize_error(e, model_name, errors, bigger_errors):
    pattern = re.compile('Observation: \{(.+)\}, Prediction: (.+), True label: (.+)')
    m = re.findall(pattern, e)
    overpredicted = m[0][1]
    underpredicted = m[0][2]
    obs = m[0][0]
    pattern = re.compile("\'w\': \'(.+)\', \'pos\'.+")
    m = re.findall(pattern, obs)
    token = m[0]
    store_error_info(errors, model_name, obs, overpredicted, token, underpredicted)
    pattern = re.compile("(.+?)_.+")
    if underpredicted != 'UNK':
        pos = re.findall(pattern,underpredicted)[0]
    else:
        pos = 'UNK'
    if not overpredicted.startswith(pos):
        store_error_info(bigger_errors, model_name, obs, overpredicted, token, underpredicted)


def store_error_info(errors, model_name, obs, overpredicted, token, underpredicted):
    if not token in errors[model_name]['token']:
        errors[model_name]['token'][token] = []
    errors[model_name]['token'][token].append({'obs': obs, 'pred': overpredicted, 'true': underpredicted})
    if not overpredicted in errors[model_name]['overpredicted']:
        errors[model_name]['overpredicted'][overpredicted] = []
    errors[model_name]['overpredicted'][overpredicted].append(
        {'obs': obs, 'pred': overpredicted, 'true': underpredicted})
    if not underpredicted in errors[model_name]['underpredicted']:
        errors[model_name]['underpredicted'][underpredicted] = []
    errors[model_name]['underpredicted'][underpredicted].append(
        {'obs': obs, 'pred': overpredicted, 'true': underpredicted})


def analyze_errors(errors, min_frequency):
    for model in errors:
        list_of_misclassified_tokens = sort_mistakes_by_len(errors, model,"token")
        list_of_underpredicted_labels = sort_mistakes_by_len(errors,model,"underpredicted")
        list_of_overpredicted_labels = sort_mistakes_by_len(errors,model,"overpredicted")
        print("Errors made by {}".format(model))
        num_errors = sum([n for n,t in list_of_misclassified_tokens])
        print("Total mistakes: {}".format(num_errors))
        #print("Wrongly classified token orthographies: {}".format(len(errors[model]['token'])))
        print("Tokens that were misclassified most times:")
        report_most_common_mistakes(list_of_misclassified_tokens, min_frequency)
        print("True labels which were missed most often:")
        report_most_common_mistakes(list_of_underpredicted_labels, min_frequency)
        print("Labels which were most often predicted instead of a true label:")
        report_most_common_mistakes(list_of_overpredicted_labels, min_frequency)
    print('Reporting differences between two models')
    for error_type in ['underpredicted','overpredicted']:
        y_true=[]
        y_pred=[]
        for i, model_name in enumerate(errors.keys()):
            model = errors[model_name]
            diffset = find_error_diff(errors, error_type, i)
            print("{} ONLY by {}:".format(error_type, model_name))
            for e in diffset:
                report = "{} ({})\t".format(e,len(model[error_type][e]))
                other = {}
                for ee in model[error_type][e]:
                    if error_type == 'underpredicted':
                        if not ee['pred'] in other:
                            other[ee['pred']] = 0
                        other[ee['pred']] += 1
                    elif error_type == 'overpredicted':
                        if not ee['true'] in other:
                            other[ee['true']] = 0
                        other[ee['true']] += 1
                    if len(model[error_type][e]) >= min_frequency:
                        y_pred.append(ee['pred'])
                        y_true.append(ee['true'])
                if error_type == "underpredicted":
                    report += 'Predicted instead: '
                elif error_type == "overpredicted":
                    report += 'The true labels were: '
                for ee in other:
                    report += "{} ({}) ".format(ee,other[ee])
                print(report)
            #if len(y_true) > 0:
                #classes = list(set(y_pred+y_true))
                #cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,labels=classes,display_labels=classes,xticks_rotation="vertical")
                #fig, ax = plt.subplots(figsize=(20,20))
                #cm.plot(ax=ax,xticks_rotation="vertical")
                #plt.savefig(model_name+'-'+error_type+'-confmatrix.png')


def plotConfusionMatrix(cm, classes, normalize=False, title='Confusion Matrix', cmap = plt.cm.Blues):

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print('Normalized Confusion Matrix')
    # else:
    #     print('Un-normalized Confusion Matrix')

    #print(cm)

    # thresh = cm.max()/2
    #
    # for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j,i, cm[i,j], horizontalalignment='center', color='white' if cm[i,j] > thresh else 'black', fontsize=25, fontweight='bold')
    #     plt.tight_layout()
    #     plt.ylabel('Actual Class')
    #     plt.xlabel('Predicted Class')
    return plt

def find_error_diff(errors, k, i):
    set1 = set()
    set2 = set()
    for j, model in enumerate(errors.keys()):
        if j == 0:
            set1 = set(errors[model][k].keys())
        elif j == 1:
            set2 = set(errors[model][k].keys())
    if set1 and set2:
        if i == 0:
            return set1 - set2
        elif i == 1:
            return set2 - set1
    else:
        return set1


def report_most_common_mistakes(list_of_mistakes, m):
    for n, t in list_of_mistakes:
        if n > m:
            print("{} ({})".format(t, n))
        else:
            break


def sort_mistakes_by_len(errors, model, key):
    list_of_mistakes = []
    for t in errors[model][key]:
        list_of_mistakes.append((len(errors[model][key][t]), t))
    list_of_mistakes = sorted(list_of_mistakes, reverse=True)
    return list_of_mistakes


with open(sys.argv[2], 'r') as f:
    errors1 = f.readlines()
errors = {sys.argv[2]: {'token':{}, 'overpredicted':{}, 'underpredicted':{}}}
bigger_errors = {sys.argv[2]: {'token': {}, 'overpredicted': {}, 'underpredicted': {}}}
for e in errors1:
    systematize_error(e,sys.argv[2],errors, bigger_errors)
if len(sys.argv) > 3:
    with open(sys.argv[3], 'r') as f:
        errors2 = f.readlines()
    errors[sys.argv[3]] = {'token':{}, 'overpredicted':{}, 'underpredicted':{}}
    bigger_errors[sys.argv[3]] = {'token': {}, 'overpredicted': {}, 'underpredicted': {}}
    for e in errors2:
        systematize_error(e,sys.argv[3],errors,bigger_errors)
min_freq = int(sys.argv[1])
analyze_errors(errors, min_freq)
print('Errors between types with different prefix:')
analyze_errors(bigger_errors,0)

