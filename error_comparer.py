import sys
import re










with open(sys.argv[1], 'r') as f:
    errors1 = f.readlines()
with open(sys.argv[2], 'r') as f:
    errors2 = f.readlines()
pattern = re.compile('Observation: \{.+\}, Prediction: (.+), True label: (.+)')
error_set1 = set()
error_set2 = set()
for e in errors1:
    error_set1.add(e)
for e in errors2:
    error_set2.add(e)

diff = error_set1 - error_set2

with open('error-diff.txt', 'w') as f:
    for e in list(diff):
        f.write(e + '\n')