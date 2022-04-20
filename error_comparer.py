import sys
import re


def systematize_error(e, model_name, errors, bigger_errors):
    pattern = re.compile('Observation: \{(.+)\}, Prediction: (.+), True label: (.+)')
    m = re.findall(pattern, e)
    overpredicted = m[0][2]
    underpredicted = m[0][1]
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
    for model in errors:
        list_of_misclassified_tokens = []
        list_of_underpredicted_labels = []
        for t in errors[model]['token']:
            list_of_misclassified_tokens.append((len(errors[model]['token'][t]),t))
        list_of_misclassified_tokens = sorted(list_of_misclassified_tokens,reverse=True)
        for l in errors[model]['underpredicted']:
            list_of_underpredicted_labels.append((len(errors[model]['underpredicted'][l]),l))
        list_of_underpredicted_labels = sorted(list_of_underpredicted_labels,reverse=True)
        print("Analysis of the performance of {}".format(model))
        print("Total mistakes: {}".format(len(errors1)))
        #print("Wrongly classified token orthographies: {}".format(len(errors[model]['token'])))
        print("Tokens that were misclassified most times:")
        for n,t in list_of_misclassified_tokens:
            if n > 3:
                print("{} ({})".format(t,n))
            else:
                break
        print("True labels which were missed most often:")
        for n,l in list_of_underpredicted_labels:
            if n > 3:
                print("{} ({})".format(l,n))

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
    error_set1.add(e)
    systematize_error(e,sys.argv[1],errors, bigger_errors)

for e in errors2:
    error_set2.add(e)
    systematize_error(e,sys.argv[2],errors,bigger_errors)

analyze_errors(errors,bigger_errors,errors1,errors2)



diff1 = error_set1 - error_set2
diff2 = error_set2 - error_set1



with open(sys.argv[1]+ '-error-diff1.txt', 'w') as f:
    for e in sorted(list(diff1)):
        f.write(e)
with open(sys.argv[2]+ 'error-diff2.txt', 'w') as f:
    for e in sorted(list(diff2)):
        f.write(e)