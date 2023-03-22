'''
These methods are for processing pydelphin-style itsdb test suites,
preparing the data for token classification based on feature vectors for scikit-learn ML library.

The output is binary files which are loadable with scikit-learn.
'''
import pickle
from delphin import itsdb
from tsdb.TestsuiteProcessor import ProcessedCorpus
from tsdb.feature_vectors import Feature_Vec_Extractor
import tsdb.pos_map
import numpy as np
from collections import OrderedDict
import glob
from tsdb import pos_map


EOS_TOKEN = 'EOS'
CONTEXT_WINDOW = 2

class Feature_Vec_Autoreg(Feature_Vec_Extractor):

    def process_testsuites(self, treebanks_path, lextypes):
        data = {'train': {}, 'dev': {}, 'test': {}}
        print('Reading test suite files into pydelphin objects...')
        for idx in ['train','dev','test']:
            for i, tsuite in enumerate(sorted(glob.iglob(treebanks_path + idx + '/**'))):
                one_corpus = self.process_one_testsuite(tsuite, idx, lextypes)
                for s in one_corpus.processed_data:
                    if len(s[0]) not in data[idx]:
                        data[idx][len(s[0])] = []
                    data[idx][len(s[0])].append((s[0], s[1]))
        return data

    def get_observations(self, terminals_tok_tags, lextypes, is_test_data):
        pos_mapper = pos_map.Pos_mapper('pos-map.txt')
        x = []
        y = []
        tokens, labels, pos_tags = \
            self.get_tokens_labels(terminals_tok_tags, CONTEXT_WINDOW, lextypes, pos_mapper)
        for k, t in enumerate(tokens):
            if k < CONTEXT_WINDOW or k >= len(tokens) - CONTEXT_WINDOW:
                continue
            y.append(labels[k])
            # For autoregressive models, pass labels
            x.append(self.get_context(t, tokens, pos_tags, k, CONTEXT_WINDOW, is_test_data, labels=labels))
        return (x,y)


    def org_sen_by_length(self, all_sentences, ts):
        n = 0
        for l in ts['sentences']:
            for s in ts['sentences'][l]:
                n += len(s)
            if l not in all_sentences:
                all_sentences[l] = []
            all_sentences[l] += ts['sentences'][l]
        return n

    def process_length(self, lextypes, items, autoregress_table, labels_table,test):
        y = []
        ys = []
        all_tokens = 0
        pos_mapper = tsdb.pos_map.Pos_mapper('pos-map.txt')  # do this for every test suite to count unknowns in each
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

    def process_table(self, data, k, lextypes, tables_by_len, test, corpus=None):
        n = 0
        for sen_len in data[k]:
            tables_by_len[k][sen_len] = {}
            autoregress_table = np.array([[{}] * len(data[k][sen_len])
                                          for i in range(sen_len)])
            labels_table = np.array([[{}] * len(data[k][sen_len]) for i in range(sen_len)])
            print("Processing sentences of length {}".format(sen_len))
            n += self.process_length(lextypes, data[k][sen_len],
                                              autoregress_table, labels_table, test=test)
            tables_by_len[k][sen_len]['ft'] = autoregress_table
            tables_by_len[k][sen_len]['lt'] = labels_table

    def get_tables_by_length(self, data, lextypes):
        tables_by_len = {'train': {}, 'dev': {}, 'test': {}}
        for k in ['train', 'dev', 'test']:
            all_tokens = 0
            test = k in ['dev', 'test']
            if test:
                for corpus in data[k]['by length']:
                    all_tokens += self.process_table(data, k, lextypes, tables_by_len, test, corpus)
            else:
                all_tokens += self.process_table(data, k, lextypes, tables_by_len, test)
            print('Total PROCESSED {} tokens: {}'.format(k, all_tokens))
        return tables_by_len

