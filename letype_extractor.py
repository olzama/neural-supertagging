from delphin import tdl, itsdb
from delphin.tokens import YYTokenLattice
import glob, sys, pathlib
import json, pickle
import numpy as np
from collections import OrderedDict
import pos_map
from datetime import datetime

CONTEXT_WINDOW = 2

DEV = ['ws212', 'ecpa']
TEST = ['cb', 'ecpr', 'jhk', 'jhu', 'tgk', 'tgu', 'psk', 'psu', #'rondane',
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
        data = {'train':{'by corpus':[], 'by length': {}},
                'test':{'by corpus':[], 'by length': {}},
                'dev':{'by corpus':[], 'by length': {}}}
        print('Reading test suite files into pydelphin objects...')
        n = 0
        for idx in ['train','dev','test']:
            t = 0
            for i, tsuite in enumerate(sorted(glob.iglob(path + idx + '/**'))):
                n += 1
                ts = itsdb.TestSuite(tsuite)
                if idx == 'train':
                    message = "A nontrain dataset {} is being added as training data!".format(ts.path.stem)
                    assert ts.path.stem not in NONTRAIN, message
                data[idx]['by corpus'].append({'name':ts.path.stem})
                items = list(ts.processed_items())
                data[idx]['by corpus'][i]['sentences'] = {}
                data[idx]['by corpus'][i]['tokens-tags'] = []
                corpus_size += len(items)
                for response in items:
                    if len(response['results']) > 0:
                        deriv = response.result(0).derivation()
                        terminals = deriv.terminals()
                        t += len(terminals)
                        p_input = response['p-input']
                        p_tokens = response['p-tokens']
                        if p_tokens:
                            terminals_tok_tags = self.map_lattice_to_input(p_input, p_tokens, deriv)
                        else:
                            terminals_tok_tags = self.get_eagle_tags(p_input,deriv)
                        if len(terminals) not in data[idx]['by corpus'][i]['sentences']:
                            data[idx]['by corpus'][i]['sentences'][len(terminals)] = []
                        data[idx]['by corpus'][i]['sentences'][len(terminals)].append(terminals_tok_tags)
                        data[idx]['by corpus'][i]['tokens-tags'].append(terminals_tok_tags)
                        if len(terminals) > max_sen_length:
                            max_sen_length = len(terminals)
            print('All raw {} tokens: {}'.format(idx,t))
            t1 = 0
            t2 = 0
            if idx == 'train':
                all_sentences = {}
                for ts in data[idx]['by corpus']:
                    t1 += self.org_sen_by_length(all_sentences, ts)
                for l in all_sentences:
                    for s in all_sentences[l]:
                        t2 += len(s)
                data[idx]['by length'] = OrderedDict(sorted(all_sentences.items()))
            else:
                for ts in data[idx]['by corpus']:
                    all_sentences = {}
                    t1 += self.org_sen_by_length(all_sentences, ts)
                    data[idx]['by length'][ts['name']] = OrderedDict(sorted(all_sentences.items()))
                for ts in data[idx]['by length']:
                    for l in data[idx]['by length'][ts]:
                        for s in data[idx]['by length'][ts][l]:
                            t2 += len(s)
            print('Added {} {} tokens to the by-corpus table'.format(t1,idx))
            print('Added {} {} tokens to the by-length table'.format(t2,idx))

        return max_sen_length, corpus_size, n+1, data

    def org_sen_by_length(self, all_sentences, ts):
        n = 0
        for l in ts['sentences']:
            for s in ts['sentences'][l]:
                n += len(s)
            if l not in all_sentences:
                all_sentences[l] = []
            all_sentences[l] += ts['sentences'][l]
        return n

    def process_testsuites_autoreg(self,testsuites,lextypes, out_dir):
        max_sen_length, corpus_size, num_ts, data = self.read_testsuites(testsuites)
        tables_by_len = {'train':{},'dev':{},'test':{}}
        for k in ['train','dev','test']:
            pathlib.Path(out_dir + '/labeled-data/' + k).mkdir(parents=True, exist_ok=False)
            all_tokens = 0
            test = k in ['dev','test']
            if test:
                for corpus in data[k]['by length']:
                    all_tokens += self.process_table(data, k, lextypes, tables_by_len, test, corpus)
            else:
                all_tokens += self.process_table(data, k, lextypes, tables_by_len, test)
            print('Total PROCESSED {} tokens: {}'.format(k, all_tokens))

    def process_testsuites_nonautoreg(self,testsuites,lextypes, out_dir):
        pos_mapper = pos_map.Pos_mapper('./pos-map.txt')
        max_sen_length, corpus_size, num_ts, data = self.read_testsuites(testsuites)
        for k in ['train','dev','test']:
            is_devtest_data = k in ['dev','test']
            pathlib.Path(out_dir + '/labeled-data/' + k).mkdir(parents=True, exist_ok=False)
            if is_devtest_data:
                for corpus in data[k]['by corpus']:
                    x,y = self.process_corpus(lextypes,corpus,pos_mapper)
                    data_table['ft'] = x
                    data_table['lt'] = y
                    with open(out_dir + '/labeled-data/' + k + '/' + corpus['name'], 'wb') as f:
                        pickle.dump(data_table, f)
            else:
                data_table = {'ft':[],'lt':[]}
                for corpus in data[k]['by corpus']:
                    x, y = self.process_corpus(lextypes,corpus,pos_mapper)
                    data_table['ft'] += x
                    data_table['lt'] += y
                with open(out_dir + '/labeled-data/train/train' , 'wb') as f:
                    pickle.dump(data_table, f)

    def process_corpus(self, lextypes, corpus,pos_mapper):
        data = []
        y = []
        for sen in corpus['tokens-tags']:
            tokens, labels, pos_tags, autoregress_labels = \
                self.get_tokens_labels(sen, CONTEXT_WINDOW, lextypes, pos_mapper, False)
            for k, t in enumerate(tokens):
                if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                    continue
                y.append(labels[k])
                data.append(self.get_context(t, tokens, pos_tags, k, CONTEXT_WINDOW))
        return data, y

    def process_table(self, data, k, lextypes, tables_by_len, test, corpus=None):
        n = 0
        table = data[k]['by length'] if not test else data[k]['by length'][corpus]
        for sen_len in table:
            tables_by_len[k][sen_len] = {}
            autoregress_table = np.array([[{}] * len(table[sen_len])
                                          for i in range(sen_len)])
            labels_table = np.array([[{}] * len(table[sen_len]) for i in range(sen_len)])
            # print("Processing sentences of length {}".format(sen_len))
            n += self.process_length(lextypes, table[sen_len],
                                              autoregress_table, labels_table, test=test)
            tables_by_len[k][sen_len]['ft'] = autoregress_table
            tables_by_len[k][sen_len]['lt'] = labels_table
        if test:
            with open(out_dir + '/labeled-data/' + k + '/' + corpus, 'wb') as f:
                pickle.dump(tables_by_len[k], f)
        else:
            with open(out_dir + '/labeled-data/train/train' , 'wb') as f:
                pickle.dump(tables_by_len[k], f)
        return n


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
                #if j % 100 == 0:
                #    print("Processing item {} out of {}...".format(j, len(items)))
                tokens,labels,pos_tags,autoregress_labels = \
                     self.get_tokens_labels(tsuite['tokens-tags'][j],CONTEXT_WINDOW, lextypes,pos_mapper,test=False)
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

    def process_length(self, lextypes, items, autoregress_table, labels_table,test):
        y = []
        ys = []
        all_tokens = 0
        pos_mapper = pos_map.Pos_mapper('./pos-map.txt')  # do this for every test suite to count unknowns in each
        for j, lst_of_terminals in enumerate(items):
            #if j % 100 == 0:
            #    print("Processing item {} out of {}...".format(j, len(items)))
            tokens,labels,pos_tags,autoregress_labels = \
                 self.get_tokens_labels(lst_of_terminals,CONTEXT_WINDOW, lextypes,pos_mapper,test)
            ys.append(labels[CONTEXT_WINDOW:CONTEXT_WINDOW*-1])
            for k, t in enumerate(tokens):
                if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                    continue
                y.append(labels[k])
                autoregress_table[k-CONTEXT_WINDOW][j] = \
                    self.get_autoregress_context(tokens,pos_tags,autoregress_labels, k,CONTEXT_WINDOW)
                labels_table[k-CONTEXT_WINDOW][j] = labels[k]
                all_tokens += 1
            y.append('\n') # sentence separator
        return all_tokens

    def leave_one_tag_per_token(self,yy_tokens):
        new_tokens = []
        prev = None
        for t in yy_tokens:
            if prev:
                if t.start != prev.start:
                    if t.end != prev.end:
                        new_tokens.append(t)
                        prev = t
            else:
                new_tokens.append(t)
                prev = t
        return new_tokens

    '''
    Sometimes, in yy_input, there are multiple possible Freeling tags for the same token.
    I don't know what the order of the tags is (may be even arbitrary).
    For now, I will just take the first tag.
    '''
    def get_eagle_tags(self, p_input, deriv):
        terminals_toks_postags = []
        yy_input = YYTokenLattice.from_string(p_input)
        if len(deriv.terminals()) != len(yy_input.tokens):
            yy_tokens = self.leave_one_tag_per_token(yy_input.tokens)
        else:
            yy_tokens = yy_input.tokens
        assert len(deriv.terminals()) == len(yy_tokens)
        for t in deriv.terminals():
            assert len(t.tokens) == 1
            toks_pos_tags = []
            for i,tok in enumerate(t.tokens):
                toks_pos_tags.append((tok, yy_tokens[i].lrules[0]))
            terminals_toks_postags.append((t,toks_pos_tags))
        return terminals_toks_postags

    def map_lattice_to_input(self, p_input, p_tokens, deriv):
        terminals_toks_postags = []
        yy_lattice = YYTokenLattice.from_string(p_tokens)
        yy_input = YYTokenLattice.from_string(p_input)
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
        return context

    def get_autoregress_context(self,tokens,pos_tags,predicted_labels, k,window):
        context = {'w':tokens[k],'pos':pos_tags[k]}
        for i in range(1,window+1):
            context['w-' + str(i)] = tokens[k-i]
            context['w+' + str(i)] = tokens[k+i]
            context['pos-' + str(i)] = pos_tags[k-i]
            context['pos+' + str(i)] = pos_tags[k+i]
            context['tag-' + str(i)] = predicted_labels[k-i] # Will be None or FAKE in test mode
        return context

    def get_tokens_labels(self, terms_and_tokens_tags, context_window, lextypes,pos_mapper, test):
        tokens = []
        labels = []
        pos_tags = []
        previous_tags = []
        for i,(terminal, toks_tags) in enumerate(terms_and_tokens_tags):
            letype = str(lextypes.get(terminal.parent.entity, "<UNK>"))
            tokens.append(terminal.form)
            labels.append(letype)
            pos_tags.append(self.get_pos_tag(toks_tags, pos_mapper))
            if test:
                previous_tags.append(None)
            else:
                previous_tags.append(letype)
        for i in range(1,1+context_window):
            tokens.insert(0, 'FAKE-' + str(i))
            labels.insert(0, 'FAKE-' + str(i))
            pos_tags.insert(0,'FAKE-' + str(i))
            previous_tags.insert(0, 'FAKE-' + str(i))
            tokens.append('FAKE+' + str(i))
            labels.append('FAKE+' + str(i))
            pos_tags.append('FAKE+' + str(i))
        return tokens, labels, pos_tags, previous_tags

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
    dt_str = '-'.join(str(datetime.now()).split()).replace(':','.')
    run_id = sys.argv[3] + dt_str
    if len(sys.argv) > 3:
        autoreg = sys.argv[4] == 'autoreg'
    out_dir = './output/' + run_id
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=False)
    le = LexTypeExtractor()
    le.parse_lexicons(args[0])
    le.stats['total lextypes'] = len(le.lextypes)
    if autoreg:
        le.process_testsuites_autoreg(args[1],le.lextypes,out_dir)
    else:
        le.process_testsuites_nonautoreg(args[1],le.lextypes,out_dir)
    with open(out_dir + '/lextypes','wb') as f:
        lextypes = set([str(v) for v in list(le.lextypes.values())])
        pickle.dump(lextypes,f)

