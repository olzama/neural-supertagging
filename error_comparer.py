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


diff1 = error_set1 - error_set2
diff2 = error_set2 - error_set1



with open(sys.argv[1]+ '-error-diff1.txt', 'w') as f:
    for e in sorted(list(diff1)):
        f.write(e)
with open(sys.argv[2]+ 'error-diff2.txt', 'w') as f:
    for e in sorted(list(diff2)):
        f.write(e)