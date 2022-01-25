from delphin import tdl, itsdb
from delphin.tokens import YYTokenLattice
import glob, sys, pathlib
import json
import re

CONTEXT_WINDOW = 2

DEV = ['ws212', 'ecpa']
TEST = ['cb', 'ecpr', 'jhk', 'jhu', 'tgk', 'tgu', 'psk', 'psu', 'rondane',
             'vm32', 'ws213', 'ws214', 'petet', 'wsj23']
IGNORE = ['ntucle', 'omw', 'wlb03', 'wnb03']
NONTRAIN = DEV + TEST + IGNORE


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
                #try:
                num_items, no_parse, sentence_lens, unk_pos = self.process_testsuite(lextypes, logf, testsuite)
                self.stats['corpora'][i]['items'] = num_items
                self.stats['corpora'][i]['noparse'] = no_parse
                self.stats['corpora'][i]['unk-pos'] = unk_pos
                all_lengths = sorted(list(sentence_lens),reverse=True)
                self.stats['corpora'][i]['max-len'] = max(all_lengths)
                # except:
                #     print("ERROR: " + testsuite)
                #     self.stats['failed corpora'].append({'name':testsuite})
                #     self.stats['corpora'].append(None)
                #     logf.write("TESTSUITE ERROR: " + testsuite + '\n')

    def process_testsuite(self, lextypes, logf, tsuite):
        ts = itsdb.TestSuite(tsuite)
        print("Processing " + ts.path.stem)
        logf.write("Processing " + ts.path.stem + '\n')
        self.stats['corpora'].append({'name': ts.path.stem})
        pairs = []
        contexts = []
        y = []
        items = list(ts.processed_items())
        noparse = 0
        unk_pos = 0
        sentence_lens = {}
        for j, response in enumerate(items):
            contexts.append([])
            if len(response['results']) > 0:
                if j % 100 == 0:
                    print("Processing item {} out of {}...".format(j, len(items)))
                result = response.result(0)
                deriv = result.derivation()
                # p_input = response['p-input']
                # p_tokens = response['p-tokens']
                # yy_lattice = YYTokenLattice.from_string(p_tokens)
                tokens,tags = \
                     self.get_tokens_tags(deriv,CONTEXT_WINDOW)
                if response['i-length'] not in sentence_lens:
                    sentence_lens[response['i-length']] = 0
                sentence_lens[response['i-length']] += 1
                for k, t in enumerate(tokens):
                    if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                        continue
                    if not t in self.stats['tokens']:
                        self.stats['tokens'][t] = 0
                    self.stats['tokens'][t] += 1
                    pairs.append((t, tags[k]))
                    y.append(tags[k])
                    contexts[j].append(self.get_context(t, tokens, tags, k, CONTEXT_WINDOW))
                pairs.append(('--EOS--','--EOS--')) # sentence separator
                y.append('\n') # sentence separator
            else:
                contexts[j].append('NO PARSE')
                noparse += 1
                err = response['error'] if response['error'] else 'None'
                #print('No parse for item {} out of {}'.format(j,len(items)))
                logf.write(ts.path.stem + '\t' + str(response['i-id']) + '\t'
                           + response['i-input'] + '\t' + err + '\n')
        self.write_output(contexts, lextypes, pairs, ts)
        return len(items), noparse, sentence_lens, unk_pos

    def write_output(self, contexts, lextypes, pairs, ts):
        for d in ['train/','test/','dev/', 'ignore/']:
            for pd in ['simple/','contexts/','true_labels/']:
                pathlib.Path('./output/' + pd + d).mkdir(parents=True, exist_ok=True)
        true_labels = []
        suf = 'train/'
        if ts.path.stem in IGNORE:
            suf = 'ignore/'
        if ts.path.stem in TEST:
            suf = 'test/'
        elif ts.path.stem in DEV:
            suf = 'dev/'
        with open('./output/simple/' + suf + ts.path.stem, 'w') as f:
            for form, entity in pairs:
                if not entity=='--EOS--':
                    letype = lextypes.get(entity, None)
                    true_labels.append(str(letype))
                    str_pair = f'{form}\t{letype}'
                    f.write(str_pair + '\n')
                else:
                    f.write('\n') # sentence separator
                    true_labels.append('\n') # sentence separator
        with open('./output/true_labels/' + suf + ts.path.stem, 'w') as f:
            for tl in true_labels:
                f.write(tl)
                if tl != '\n':
                    f.write('\n')
        with open('./output/contexts/' + suf + ts.path.stem, 'w') as f:
            f.write(json.dumps(contexts))

    def get_context(self, t, tokens,tags, i, window):
        context = {'w': t}
        for j in range(1,window+1):
            prev_tok = tokens[i-j]
            prev_tag = tags[i-j]
            next_tok = tokens[i+j]
            context['w-' + str(j)] = prev_tok
            context['w+' + str(j)] = next_tok
            context['tag-' + str(j)] = prev_tag
        return context

    # Work in progress
    # def get_tokens_tags(self, deriv, context_window, p_input, yy_lattice, sentence, unk_pos, logf):
    #     tokens = []
    #     pos_tags = []
    #     tags = []
    #     p_input_per_word = [m.group() for m in re.finditer('\([^()]+\)',p_input)]
    #     for ptok in p_input_per_word:
    #         items = ptok.split(',')
    #         span = items[3].strip()
    #         match = re.search('<([0-9]+):([0-9]+)>',span)
    #         start = match.group(1)
    #         end = match.group(2)
    #         for yyt in yy_lattice.tokens:
    #             if yyt[3].data[0] == start and yyt[3].data[1] == end:
    #                 print(yyt)
    #         print(ptok)
    #     for i,terminal in enumerate(deriv.terminals()):
    #         tokens.append(terminal.form)
    #         tags.append(terminal.parent.entity)
    #         try:
    #             pos_tag = self.get_pos_tag(p_input_per_word[i], terminal.form, sentence, logf, unk_pos)
    #             pos_tags.append(pos_tag)
    #         except:
    #             print(i)
    #     for i in range(1,1+context_window):
    #         tokens.insert(0, 'FAKE-' + str(i))
    #         tags.insert(0, 'FAKE-' + str(i))
    #         pos_tags.insert(0,'FAKE-'+str(i))
    #         tokens.append('FAKE+' + str(i))
    #         tags.append('FAKE+' + str(i))
    #         pos_tags.append('FAKE+' + str(i))
    #     return tokens, tags, pos_tags

    def get_tokens_tags(self, deriv, context_window):
        tokens = []
        tags = []
        for i,terminal in enumerate(deriv.terminals()):
            tokens.append(terminal.form)
            tags.append(terminal.parent.entity)
        for i in range(1,1+context_window):
            tokens.insert(0, 'FAKE-' + str(i))
            tags.insert(0, 'FAKE-' + str(i))
            tokens.append('FAKE+' + str(i))
            tags.append('FAKE+' + str(i))
        return tokens, tags


    def get_pos_tag(self, p_input,token, sentence,logf,unk_pos_tags):
        items = p_input.split(',')
        try:
            pos_tag,prob = items[-1].strip().split(' ')
        except:
            pos_tag = '"<UNK-POS>"'
            logf.write('Missing POS tag for token {} in sentence {}'.format(token, sentence))
            unk_pos_tags += 1
        pos_tag = pos_tag.strip('"')
        return pos_tag

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

