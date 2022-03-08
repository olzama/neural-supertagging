from delphin import tdl, itsdb
from delphin.tokens import YYTokenLattice
import glob, sys, pathlib
import json, pickle
import numpy as np
from collections import OrderedDict
import pos_map

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

    def read_testsuites(self,path):
        max_sen_length = 0
        corpus_size = 0
        data = {'train':{'by corpus':[], 'by length':OrderedDict()},
                'test':{'by corpus':[], 'by length':OrderedDict()},
                'dev':{'by corpus':[], 'by length':OrderedDict()}}
        print('Reading test suite files into pydelphin objects...')
        for i, tsuite in enumerate(sorted(glob.iglob(path + '**'))):
            ts = itsdb.TestSuite(tsuite)
            if ts.path.stem in TEST:
                idx = 'test'
            elif ts.path.stem in DEV:
                idx = 'dev'
            elif ts.path.stem not in IGNORE:
                idx = 'train'
            data[idx]['by corpus'].append({'name':ts.path.stem})
            items = list(ts.processed_items())
            data[idx]['by corpus'][i]['sentences'] = {}
            data[idx]['by corpus'][i]['tokens-tags'] = []
            corpus_size += len(items)
            for response in items:
                if len(response['results']) > 0:
                    deriv = response.result(0).derivation()
                    terminals = deriv.terminals()
                    p_input = response['p-input']
                    p_tokens = response['p-tokens']
                    terminals_tok_tags = self.map_lattice_to_input(p_input, p_tokens, deriv)
                    if len(terminals) not in data[idx]['by corpus'][i]['sentences']:
                        data[idx]['by corpus'][i]['sentences'][len(terminals)] = []
                    data[idx]['by corpus'][i]['sentences'][len(terminals)].append(terminals_tok_tags)
                    data[idx]['by corpus'][i]['tokens-tags'].append(terminals_tok_tags)
                    if len(terminals) > max_sen_length:
                        max_sen_length = len(terminals)
        all_train_sentences = {}
        for ts in data['train']['by corpus']:
            all_train_sentences.update(ts['sentences'])
        data['train']['by length'] = OrderedDict(sorted(all_train_sentences.items()))
        return max_sen_length, corpus_size, i+1, data

    def process_testsuites(self,testsuites,lextypes):
        max_sen_length, corpus_size, num_ts, data = self.read_testsuites(testsuites)
        train_tables_by_len = {}
        with open('./log.txt', 'w') as logf:
            for k in ['train','dev','test']:
                start = 0
                autoregress_table = np.array([[{}] * corpus_size for i in range(max_sen_length)])
                labels_table = np.array([[{}] * corpus_size for i in range(max_sen_length)])
                for testsuite in data[k]['by corpus']:
                    true_labels = self.process_testsuite(
                        lextypes, logf, testsuite, autoregress_table, labels_table, start)
                    testsuite['column'] = start
                    testsuite['true labels'] = true_labels
                    start = start + len(testsuite['sentences'])
                with open('./output/by-corpus/autoregressive/feature_table-'+k, 'wb') as f:
                    pickle.dump(autoregress_table, f)
                with open('./output/by-corpus/autoregressive/labels_table-'+k, 'wb') as f:
                    pickle.dump(labels_table, f)
                with open('./output/by-corpus/data/data-'+k,'wb') as f:
                    pickle.dump(data[k]['by corpus'],f)
                for sen_len in data['train']['by length']:
                    train_tables_by_len[sen_len] = {}
                    autoregress_table = np.array([[{}] * len(data['train']['by length'][sen_len])
                                                  for i in range(sen_len)])
                    labels_table = np.array([[{}] * len(data['train']['by length'][sen_len]) for i in range(sen_len)])
                    print("Processing sentences of length {}".format(sen_len))
                    logf.write("Processing sentences of length {}\n".format(sen_len))
                    true_labels = self.process_length(lextypes, data['train']['by length'][sen_len],
                                                      autoregress_table,labels_table)
                    train_tables_by_len[sen_len]['ft'] = autoregress_table
                    train_tables_by_len[sen_len]['lt'] = labels_table
            return train_tables_by_len


    '''
    Assume a numpy table coming in. Get e.g. tokens 2 through 5 in sentences 4 and 5,
    for the test suite #20 in the data.
    '''
    def get_table_portion(self, ts_info, table, ts_num, token_range, sentence_range):
        ts_column = ts_info[ts_num]['column']
        tokens = sum(ts_info[ts_num]['sentences'][sentence_range[0]:sentence_range[1]])
        return table[token_range[0]:token_range[1],ts_column:ts_column+tokens]


    def process_testsuite(self, lextypes, logf, tsuite, autoregress_table, labels_table, start):
        print("Processing " + tsuite['name'])
        logf.write("Processing " + tsuite['name'] + '\n')
        pairs = []
        contexts = []
        y = []
        ys = []
        pos_mapper = pos_map.Pos_mapper('./pos-map.txt')  # do this for every test suite to count unknowns in each
        for sentence_len in tsuite['sentences']:
            items = tsuite['sentences'][sentence_len]
            for j, lst_of_terminals in enumerate(items):
                contexts.append([])
                if j % 100 == 0:
                    print("Processing item {} out of {}...".format(j, len(items)))
                tokens,labels,pos_tags,autoregress_labels = \
                     self.get_tokens_labels(tsuite['tokens-tags'][j],CONTEXT_WINDOW, lextypes,pos_mapper)
                ys.append(labels[CONTEXT_WINDOW:CONTEXT_WINDOW*-1])
                for k, t in enumerate(tokens):
                    if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                        continue
                    pairs.append((t, labels[k]))
                    y.append(labels[k])
                    contexts[j].append(self.get_context(t, tokens, pos_tags, k, CONTEXT_WINDOW))
                    autoregress_table[k-CONTEXT_WINDOW][start+j] = \
                        self.get_autoregress_context(tokens,pos_tags,autoregress_labels, k,CONTEXT_WINDOW)
                    labels_table[k-CONTEXT_WINDOW][start+j] = labels[k]
                pairs.append(('--EOS--','--EOS--')) # sentence separator
                y.append('\n') # sentence separator
        self.write_output(contexts, pairs, tsuite['name'])
        return ys

    def process_length(self, lextypes, items, autoregress_table, labels_table):
        y = []
        ys = []
        pos_mapper = pos_map.Pos_mapper('./pos-map.txt')  # do this for every test suite to count unknowns in each
        for j, lst_of_terminals in enumerate(items):
            if j % 100 == 0:
                print("Processing item {} out of {}...".format(j, len(items)))
            tokens,labels,pos_tags,autoregress_labels = \
                 self.get_tokens_labels(lst_of_terminals,CONTEXT_WINDOW, lextypes,pos_mapper)
            ys.append(labels[CONTEXT_WINDOW:CONTEXT_WINDOW*-1])
            for k, t in enumerate(tokens):
                if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                    continue
                y.append(labels[k])
                autoregress_table[k-CONTEXT_WINDOW][j] = \
                    self.get_autoregress_context(tokens,pos_tags,autoregress_labels, k,CONTEXT_WINDOW)
                labels_table[k-CONTEXT_WINDOW][j] = labels[k]
            y.append('\n') # sentence separator
        return ys

    def map_lattice_to_input(self, p_input, p_tokens, deriv):
        yy_lattice = YYTokenLattice.from_string(p_tokens)
        yy_input = YYTokenLattice.from_string(p_input)
        terminals_toks_postags = []
        for t in deriv.terminals():
            toks_pos_tags = []
            for ttok in t.tokens:
                span = None
                pos_probs = {}
                for lat_tok in yy_lattice.tokens:
                    if lat_tok.id == ttok.id:
                        span = lat_tok.lnk.data
                        break
                for i,in_tok in enumerate(yy_input.tokens):
                    if in_tok.lnk.data[0] == span[0]:
                        for pos, p in in_tok.pos:
                            if pos not in pos_probs:
                                pos_probs[pos] = []
                            pos_probs[pos].append(float(p))
                        if in_tok.lnk.data[1] != span[1]:
                            cur_tok = in_tok
                            while cur_tok.lnk.data[1] != span[1]:
                                next_tok = yy_input.tokens[i+1]
                                i += 1
                                for pos, p in next_tok.pos:
                                    if pos not in pos_probs:
                                        pos_probs[pos] = []
                                    pos_probs[pos].append(float(p))
                                cur_tok = next_tok
                        else:
                            break
                toks_pos_tags.append((ttok, pos_probs))
            terminals_toks_postags.append((t,toks_pos_tags))
        return terminals_toks_postags

    def write_output(self, contexts, pairs, ts_name):
        for d in ['train/','test/','dev/', 'ignore/']:
            for pd in ['simple/','by-corpus/contexts/','by-corpus/true_labels/']:
                pathlib.Path('./output/' + pd + d).mkdir(parents=True, exist_ok=True)
        true_labels = []
        suf = 'train/'
        if ts_name in IGNORE:
            suf = 'ignore/'
        if ts_name in TEST:
            suf = 'test/'
        elif ts_name in DEV:
            suf = 'dev/'
        with open('./output/simple/' + suf + ts_name, 'w') as f:
            for form, letype in pairs:
                if not letype=='--EOS--':
                    true_labels.append(str(letype))
                    str_pair = f'{form}\t{letype}'
                    f.write(str_pair + '\n')
                else:
                    f.write('\n') # sentence separator
                    true_labels.append('\n') # sentence separator
        with open('./output/by-corpus/true_labels/' + suf + ts_name, 'w') as f:
            for tl in true_labels:
                f.write(tl)
                if tl != '\n':
                    f.write('\n')
        with open('./output/by-corpus/contexts/' + suf + ts_name, 'w') as f:
            f.write(json.dumps(contexts))

    def get_context(self, t, tokens, pos_tags, i, window):
        context = {'w': t, 'pos': pos_tags[i]}
        for j in range(1,window+1):
            prev_tok = tokens[i-j]
            prev_pos = pos_tags[i-j]
            next_tok = tokens[i+j]
            next_pos = pos_tags[i+j]
            context['w-' + str(j)] = prev_tok
            context['w+' + str(j)] = next_tok
            context['pos-' + str(j)] = prev_pos
            context['pos+' + str(j)] = next_pos
            #context['tag-' + str(j)] = prev_tag # this is a bug: can't use gold tags. Need an autoregressive model.
        return context

    def get_autoregress_context(self,tokens,pos_tags,predicted_labels, k,window):
        context = {'w':tokens[k]} #,'pos':pos_tags[k]}
        for i in range(1,window+1):
            #context['w-' + str(i)] = tokens[k-i]
            #context['w+' + str(i)] = tokens[k+i]
            context['tag-' + str(i)] = predicted_labels[k-i] # Will be None or FAKE
        return context


    def get_tokens_labels(self, terms_and_tokens_tags, context_window, lextypes,pos_mapper):
        tokens = []
        labels = []
        pos_tags = []
        predicted_labels = []
        for i,(terminal, toks_tags) in enumerate(terms_and_tokens_tags):
            letype = lextypes.get(terminal.parent.entity, "<UNK>")
            tokens.append(terminal.form)
            labels.append(str(letype))
            pos_tags.append(self.get_pos_tag(toks_tags, pos_mapper))
            predicted_labels.append(None) # for autoregressive models
        for i in range(1,1+context_window):
            tokens.insert(0, 'FAKE-' + str(i))
            labels.insert(0, 'FAKE-' + str(i))
            pos_tags.insert(0,'FAKE-' + str(i))
            predicted_labels.insert(0, 'FAKE-' + str(i))
            tokens.append('FAKE+' + str(i))
            labels.append('FAKE+' + str(i))
            pos_tags.append('FAKE+' + str(i))
        return tokens, labels, pos_tags, predicted_labels

    def get_pos_tag(self,tokens_tags, pos_mapper):
        tag = ''
        for tt in tokens_tags:
            pos_probs = tt[1]
            for pos in pos_probs:
                tag = tag + '+' + pos
        tag = tag.strip('+')
        if '+' in tag:
            tag = pos_mapper.map_tag(tag)
        return tag

if __name__ == "__main__":
    args = sys.argv[1:]
    le = LexTypeExtractor()
    le.parse_lexicons(args[0])
    le.stats['total lextypes'] = len(le.lextypes)
    train_tables = le.process_testsuites(args[1],le.lextypes)
    with open('./output/by-length/tables_by_length', 'wb') as f:
        pickle.dump(train_tables,f)
    with open('./output/lextypes','wb') as f:
        pickle.dump(set([str(v) for v in list(le.lextypes.values())]),f)

