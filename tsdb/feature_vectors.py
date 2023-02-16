'''
These methods are for processing pydelphin-style itsdb test suites,
preparing the data for token classification based on feature vectors for scikit-learn ML library.

The output is binary files which are loadable with scikit-learn.
'''

import pathlib
import pickle
from delphin import itsdb
from delphin.tokens import YYTokenLattice
from tsdb.TestsuiteProcessor import ProcessedCorpus, TestsuiteProcessor
import pos_map

EOS_TOKEN = 'EOS'
CONTEXT_WINDOW = 2

class Feature_Vec_Extractor(TestsuiteProcessor):

    def __init__(self):
        pass

    @property
    def output_format(self):
        return 'wb'

    def process_one_testsuite(self, tsuite, type, lextypes):
        ts = itsdb.TestSuite(tsuite)
        all_sentences = []
        parsed_sentences = []
        items = list(ts.processed_items())
        observations = []
        total_tokens = 0
        print("{} sentences in corpus {} including possible sentences with no parse.".format(len(items), ts.path.stem))
        for response in items:
            all_sentences.append(response['i-input'])
            if len(response['results']) > 0:
                parsed_sentences.append(response['i-input'])
                deriv = response.result(0).derivation()
                p_input = response['p-input']
                p_tokens = response['p-tokens']
                terminals_tok_tags = self.map_lattice_to_input(p_input, p_tokens, deriv)
                observations.append(self.get_observations(terminals_tok_tags, lextypes))
        pc = ProcessedCorpus(ts.path.stem, type, observations, all_sentences, parsed_sentences, total_tokens )
        return pc

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

    '''
    Input: a list of pydelphin itsdb terminals and a list of known lexical types (e.g. from the training data).
    Output: a list of tuples: the terminal orthographic form and its lexical type, if it is known, or <UNK> otherwise.
    '''
    def get_observations(self, terminals_tok_tags, lextypes):
        pos_mapper = pos_map.Pos_mapper('pos-map.txt')
        x = []
        y = []
        tokens, labels, pos_tags, autoregress_labels = \
            self.get_tokens_labels(terminals_tok_tags, CONTEXT_WINDOW, lextypes, pos_mapper, False)
        for k, t in enumerate(tokens):
            if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                continue
            y.append(labels[k])
            x.append(self.get_context(t, tokens, pos_tags, k, CONTEXT_WINDOW))
        return (x,y)

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

    def write_out_one_corpus(self, f, pc, total_sen, total_tok):
        data = {'ft': [], 'lt': []}
        for x,y in pc.processed_data:
            data['ft'].append(x)
            data['lt'].append(y)
            total_tok += len(x)
            total_sen += 1
        pickle.dump(data,f)
        return total_sen, total_tok

