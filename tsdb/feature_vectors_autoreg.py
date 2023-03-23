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

    def write_output_by_split(self, dest_path, data):
        print('Writing output to {}'.format(dest_path))
        for split_type in ['train', 'dev', 'test']:
            with open(dest_path + split_type + '/' + split_type, self.output_format) as f:
                whole_output = {}
                total_sen = 0
                total_tok = 0
                for sen_len in data[split_type]:
                    whole_output[sen_len] = {'ft': [], 'lt': []}
                    for vec in data[split_type][sen_len]:
                        whole_output[sen_len]['ft'].extend(vec[0])
                        whole_output[sen_len]['lt'].extend(vec[1])
                        total_sen += 1
                        total_tok += len(vec[0])
                pickle.dump(whole_output,f)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, split_type))

    def write_output_by_corpus(self, dest_path, data):
        # TODO: Need to either reimplement this method or make it illegal? throw an exception?
        pass

