from delphin import tdl, itsdb
import glob, sys, pathlib
import json

CONTEXT_WINDOW = 2

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
            for i,testsuite in enumerate(glob.iglob(testsuites+'**')):
                try:
                    num_items, no_parse = self.process_testsuite(lextypes, logf, testsuite)
                    self.stats['corpora'][i]['items'] = num_items
                    self.stats['corpora'][i]['noparse'] = no_parse
                except:
                    print("ERROR: " + testsuite)
                    self.stats['failed corpora'].append({'name':testsuite})
                    self.stats['corpora'].append(None)
                    logf.write("TESTSUITE ERROR: " + testsuite + '\n')

    def process_testsuite(self, lextypes, logf, tsuite):
        ts = itsdb.TestSuite(tsuite)
        print("Processing " + ts.path.stem)
        logf.write("Processing " + ts.path.stem + '\n')
        self.stats['corpora'].append({'name': ts.path.stem})
        pairs = []
        tag_windows = []
        items = list(ts.processed_items())
        noparse = 0
        for j, response in enumerate(items):
            if len(response['results']) > 0:
                if j % 100 == 0:
                    print("Processing item {} out of {}...".format(j, len(items)))
                result = response.result(0)
                deriv = result.derivation()
                tokens,tags = self.get_tokens_tags(deriv,CONTEXT_WINDOW)
                for k, t in enumerate(tokens):
                    if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                        continue
                    if not t in self.stats['tokens']:
                        self.stats['tokens'][t] = 0
                    self.stats['tokens'][t] += 1
                    pairs.append((t, tags[k]))
                    tag_windows.append(self.get_tag_window(t,tokens,tags,k,CONTEXT_WINDOW))

            else:
                noparse += 1
                err = response['error'] if response['error'] else 'None'
                #print('No parse for item {} out of {}'.format(j,len(items)))
                logf.write(ts.path.stem + '\t' + str(response['i-id']) + '\t'
                           + response['i-input'] + '\t' + err + '\n')
        pathlib.Path('./output/simple/').mkdir(parents=True, exist_ok=True)
        pathlib.Path('./output/contexts/').mkdir(parents=True, exist_ok=True)
        with open('./output/simple/' + ts.path.stem + '-simple-seq.txt', 'w') as f:
            for form, entity in pairs:
                letype = lextypes.get(entity, None)
                str_pair = f'{form}\t{letype}'
                f.write(str_pair + '\n')
        with open('./output/contexts/' + ts.path.stem + '-windows.txt', 'w') as f:
            js = json.dumps(tag_windows)
            f.write(js)
        return len(items), noparse

    def get_tag_window(self,t,tokens,tags,i,window):
        tag_window = {'w':t,'tag':tags[i]}
        for j in range(1,window+1):
            prev_tok = tokens[i-j]
            prev_tag = tags[i-j]
            next_tok = tokens[i+j]
            tag_window['w-' + str(j)] = prev_tok
            tag_window['w+' + str(j)] = next_tok
            tag_window['tag-' + str(j)] = prev_tag
        return tag_window

    def get_tokens_tags(self, deriv, context_window):
        tokens = []
        tags = []
        for terminal in deriv.terminals():
            tokens.append(terminal.form)
            tags.append(terminal.parent.entity)
        for i in range(1,1+context_window):
            tokens.insert(0, 'FAKE-' + str(i))
            tags.insert(0, 'FAKE-' + str(i))
            tokens.append('FAKE+' + str(i))
            tags.append('FAKE+' + str(i))
        return tokens,tags


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
