from delphin import tdl, itsdb
from delphin.tokens import YYTokenLattice
import glob, sys, pathlib
import json
import os

from pos_map import POS_MAP

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
        mwe = {}
        with open('./log.txt', 'w') as logf:
            for i,testsuite in enumerate(glob.iglob(testsuites+'**')):
                #try:
                num_items, no_parse, sentence_lens, unk_pos = self.process_testsuite(lextypes, logf, testsuite, mwe)
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
        with open('./mwe.txt', 'w') as f:
            for tag in mwe:
                f.write(tag)
                for form in mwe[tag]:
                    f.write('\t' + form)
                f.write('\n')


    def process_testsuite(self, lextypes, logf, tsuite, mwe):
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
                p_input = response['p-input']
                p_tokens = response['p-tokens']
                terminals_toks_pos_tags = self.map_lattice_to_input(p_input,p_tokens, deriv)
                tokens,tags,pos_tags = \
                     self.get_tokens_tags(terminals_toks_pos_tags,CONTEXT_WINDOW, lextypes,mwe)
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
        self.write_output(contexts, pairs, ts)
        return len(items), noparse, sentence_lens, unk_pos

    def map_lattice_to_input(self, p_input,p_tokens, deriv):
        yy_lattice = YYTokenLattice.from_string(p_tokens)
        yy_input = YYTokenLattice.from_string(p_input)
        terminals = deriv.terminals()
        terminals_toks_postags = []
        for t in terminals:
            toks_pos_tags = []
            for ttok in t.tokens:
                span = None
                id = ttok.id
                pos_probs = {}
                for lat_tok in yy_lattice.tokens:
                    if lat_tok.id == id:
                        span = lat_tok.lnk.data
                        break
                for i,in_tok in enumerate(yy_input.tokens):
                    if in_tok.lnk.data[0] == span[0]:
                        for pos,p in in_tok.pos:
                            if pos not in pos_probs:
                                pos_probs[pos] = []
                            pos_probs[pos].append(float(p))
                        if in_tok.lnk.data[1] != span[1]:
                            cur_tok = in_tok
                            while cur_tok.lnk.data[1] != span[1]:
                                next_tok = yy_input.tokens[i+1]
                                i += 1
                                for pos,p in next_tok.pos:
                                    if pos not in pos_probs:
                                        pos_probs[pos] = []
                                    pos_probs[pos].append(float(p))
                                cur_tok = next_tok
                        else:
                            break
                toks_pos_tags.append((ttok, pos_probs))
            terminals_toks_postags.append((t,toks_pos_tags))
        return terminals_toks_postags


    def write_output(self, contexts, pairs, ts):
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
            for form, letype in pairs:
                if not letype=='--EOS--':
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


    def get_tokens_tags(self, terms_and_tokens_tags, context_window, lextypes,mwe):
        tokens = []
        tags = []
        pos_tags = []
        for i,(terminal, toks_tags) in enumerate(terms_and_tokens_tags):
            letype = lextypes.get(terminal.parent.entity, None)
            tokens.append(terminal.form)
            tags.append(letype)
            pos_tags.append(self.get_pos_tag(toks_tags, terminal.form,mwe))
        for i in range(1,1+context_window):
            tokens.insert(0, 'FAKE-' + str(i))
            tags.insert(0, 'FAKE-' + str(i))
            tokens.append('FAKE+' + str(i))
            tags.append('FAKE+' + str(i))
        return tokens, tags, pos_tags

    def get_pos_tag(self,tokens_tags, form, mwe):
        tag = ''
        for tt in tokens_tags:
            pos_probs = tt[1]
            for pos in pos_probs:
                #print(form + '\t' + pos + '\t')
                tag = tag + '+' + pos
        tag = tag.strip('+')
        if '+' in tag:
            if not tag in mwe:
                mwe[tag] = set()
            mwe[tag].add(form)
        return tag

    def map_tag_sequence(self,seq):
        pass

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

