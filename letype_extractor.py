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
        return lextypes

    def process_testsuites(self,testsuites,lextypes):
        with open('./log.txt', 'w') as logf:
            for j,testsuite in enumerate(glob.iglob(testsuites+'**')):
                try:
                    ts = itsdb.TestSuite(testsuite)
                except:
                    print("ERROR: " + ts.path.stem)
                    self.stats['failed corpora'].append({'name':ts.path.stem})
                    logf.write("TESTSUITE ERROR: " + ts.path.stem + '\n')
                print("Processing " + ts.path.stem)
                self.stats['corpora'].append({'name':ts.path.stem})
                pairs = []
                items = list(ts.processed_items())
                self.stats['corpora'][j]['items'] = len(items)
                self.stats['corpora'][j]['noparse'] = 0
                for i,response in enumerate(items):
                    if len(response['results']) > 0:
                        if i%100==0:
                            print("Processing item {} out of {}...".format(i,len(items)))
                        result = response.result(0)
                        deriv = result.derivation()
                        for t in deriv.terminals():
                            if not t.form in self.stats['tokens']:
                                self.stats['tokens'][t.form] = 0
                            self.stats['tokens'][t.form] += 1
                            pairs.append((t.form, t.parent.entity))
                    else:
                        self.stats['corpora'][j]['noparse'] += 1
                        err = response['error'] if response['error'] else 'None'
                        logf.write(ts.path.stem + '\t' + str(response['i-id']) + '\t'
                                   + response['i-input'] + '\t' + err +'\n')
                with open('./output/'+ts.path.stem+'.txt', 'w') as f:
                    for form, entity in pairs:
                        letype = lextypes.get(entity, None)
                        str_pair = f'{form}\t{letype}'
                        f.write(str_pair+'\n')



if __name__ == "__main__":
    args = sys.argv[1:]
    le = LexTypeExtractor()
    lextypes = le.parse_lexicons(args[0])
    le.stats['total lextypes'] = len(lextypes)
    le.process_testsuites(args[1],lextypes)
    with open('stats.txt','w') as f:
        for c in le.stats['corpora']:
            for item in c:
                f.write(str(item) + ': ' + str(c[item]) + '\n')
        f.write('Failed to load corpora:' + str(len(le.stats['failed corpora'])) + '\n')
        for fc in le.stats['failed corpora']:
            f.write(fc + '\n')
        f.write('Total tokens in all corpora: ' + str(sum(le.stats['tokens'].values())) + '\n')
        f.write('Total unique tokens: ' + str(len(le.stats['tokens'])) + '\n')
        f.write('Total lextypes: ' + str(le.stats['total lextypes']) + '\n')
