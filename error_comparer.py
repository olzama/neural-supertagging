import sys
import re
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



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
    pattern = re.compile("(.+?_.+?)_.+")
    pos = re.findall(pattern,underpredicted)[0]
    # Next: why does the above not capture what I want?
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


def analyze_errors(errors, bigger_errors, error_list1, error_list2):
    min_frequency = 3
    for model in errors:
        list_of_misclassified_tokens = sort_mistakes_by_len(errors, model,"token")
        list_of_underpredicted_labels = sort_mistakes_by_len(errors,model,"underpredicted")
        list_of_overpredicted_labels = sort_mistakes_by_len(errors,model,"overpredicted")
        print("Errors made by {}".format(model))
        print("Total mistakes: {}".format(len(errors1)))
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
            print("{} by {}:".format(error_type, model_name))
            for e in diffset:
                print("{} ({})".format(e,len(model[error_type][e])))
                for ee in errors[model][error_type][e]:
                    if error_type == 'underpredicted':
                        print('Predicted: {}'.format(ee['pred']))
                    elif error_type == 'overpredicted':
                        print('True label: {}'.format(ee['true']))
                    y_pred.append(ee['pred'])
                    y_true.append(ee['true'])
            cm = confusion_matrix(y_true,y_pred)
            ConfusionMatrixDisplay(cm).plot()
            plt.savefig(model_name+'-confmatrix.png')



def find_error_diff(errors, k, i):
    for j, model in enumerate(errors.keys()):
        if j == 0:
            set1 = set(errors[model][k].keys())
        elif j == 1:
            set2 = set(errors[model][k].keys())
    if i == 0:
        return set1 - set2
    elif i == 1:
        return set2 - set1


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


with open(sys.argv[1], 'r') as f:
    errors1 = f.readlines()
with open(sys.argv[2], 'r') as f:
    errors2 = f.readlines()
error_set1 = set()
error_set2 = set()
bigger_error_set1 = set()
bigger_error_set2 = set()
errors = {sys.argv[1]: {'token':{}, 'overpredicted':{}, 'underpredicted':{}},
          sys.argv[2]: {'token':{}, 'overpredicted':{}, 'underpredicted':{}}}

bigger_errors = {sys.argv[1]: {'token':{}, 'overpredicted':{}, 'underpredicted':{}},
                 sys.argv[2]: {'token':{}, 'overpredicted':{}, 'underpredicted':{}}}

for e in errors1:
    systematize_error(e,sys.argv[1],errors, bigger_errors)

for e in errors2:
    systematize_error(e,sys.argv[2],errors,bigger_errors)

analyze_errors(errors,bigger_errors,errors1,errors2)






with open(sys.argv[1]+ '-error-diff1.txt', 'w') as f:
    for e in sorted(list(diff1)):
        f.write(e)
with open(sys.argv[2]+ 'error-diff2.txt', 'w') as f:
    for e in sorted(list(diff2)):
        f.write(e)