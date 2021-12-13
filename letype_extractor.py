from delphin import tdl, itsdb
import glob, sys


class LexTypeExtractor:
    def __init__(self):
        self.stats = {'corpora': [], 'failed corpora': [], 'tokens': {}, 'total lextypes': 0}

    def parse_lexicons(self,lexicons):
        lextypes = {}  # mapping of lexical entry IDs to types
        for lexicon in glob.iglob(lexicons+'**'):
            for event, obj, lineno in tdl.iterparse(lexicon):
                if event == 'TypeDefinition':
                    lextypes[obj.identifier] = obj.supertypes[0]  # assume exactly 1
        self.lextypes = lextypes

    def process_testsuites(self,testsuites,lextypes):
        with open('./log.txt', 'w') as logf:
            for j,testsuite in enumerate(glob.iglob(testsuites+'**')):
                try:
                    num_items, no_parse = self.process_testsuite(j, lextypes, logf, testsuite)
                    self.stats['corpora'][j]['items'] = num_items
                    self.stats['corpora'][j]['noparse'] = no_parse
                except:
                    print("ERROR: " + testsuite)
                    self.stats['failed corpora'].append({'name':testsuite})
                    self.stats['corpora'].append(None)
                    logf.write("TESTSUITE ERROR: " + testsuite + '\n')

    def process_testsuite(self, i, lextypes, logf, tsuite):
        ts = itsdb.TestSuite(tsuite)
        print("Processing " + ts.path.stem)
        self.stats['corpora'].append({'name': ts.path.stem})
        pairs = []
        items = list(ts.processed_items())
        noparse = 0
        for j, response in enumerate(items):
            if len(response['results']) > 0:
                if j % 100 == 0:
                    print("Processing item {} out of {}...".format(j, len(items)))
                result = response.result(0)
                deriv = result.derivation()
                for t in deriv.terminals():
                    if not t.form in self.stats['tokens']:
                        self.stats['tokens'][t.form] = 0
                    self.stats['tokens'][t.form] += 1
                    pairs.append((t.form, t.parent.entity))
            else:
                noparse += 1
                err = response['error'] if response['error'] else 'None'
                logf.write(ts.path.stem + '\t' + str(response['i-id']) + '\t'
                           + response['i-input'] + '\t' + err + '\n')
        with open('./output/' + ts.path.stem + '.txt', 'w') as f:
            for form, entity in pairs:
                letype = lextypes.get(entity, None)
                str_pair = f'{form}\t{letype}'
                f.write(str_pair + '\n')
        return len(items), noparse


if __name__ == "__main__":
    args = sys.argv[1:]
    le = LexTypeExtractor()
    le.parse_lexicons(args[0])
    le.stats['total lextypes'] = len(le.lextypes)
    le.process_testsuites(args[1],le.lextypes)
    with open('stats.txt','w') as f:
        for c in le.stats['corpora']:
            # This will print out the name of the corpus if the corpus was successfully loaded
            if c:
                for item in c:
                    f.write(str(item) + ': ' + str(c[item]) + '\n')
        f.write('Failed to load corpora:' + str(len(le.stats['failed corpora'])) + '\n')
        for fc in le.stats['failed corpora']:
            f.write(fc['name'] + '\n')
        f.write('Total tokens in all corpora: ' + str(sum(le.stats['tokens'].values())) + '\n')
        f.write('Total unique tokens: ' + str(len(le.stats['tokens'])) + '\n')
        f.write('Total lextypes: ' + str(le.stats['total lextypes']) + '\n')
