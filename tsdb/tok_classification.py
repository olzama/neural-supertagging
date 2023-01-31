'''
These methods are for processing pydelphin-style itsdb test suites,
preparing the data for token classification based on simple token-tag pairs, tab-separated.

The output format for the first three sentences of the MRS testsuite is:

it	n_-_pr-it-x_le
rained	v_-_it_le
.	pt_-_period_le

abrams	n_-_pn_le
barked	v_-_le
.	pt_-_period_le

the	d_-_the_le
window	n_-_c-ed-ns_le
opened	v_pp_unacc_le
.	pt_-_period_le

'''


import glob

from delphin import itsdb

from letype_extractor import ProcessedCorpus
from TestsuiteProcessor import TestsuiteProcessor

class Token_Tag_Extractor(TestsuiteProcessor):

    def __init__(self):
        pass

    '''
    Takes a path to treebanks, already separated into train, dev, and test folders.
    The treebanks are in [incr tsdb()] format.
    Outputs a dictionary of three ProcessedCorpus lists (train, dev, and test).
    Each ProcessedCorpus contains a list of token-tag pairs.
    '''
    def process_testsuites(self, treebanks_path, lextypes):
        data = {'train': [], 'dev': [], 'test': []}
        print('Reading test suite files into pydelphin objects...')
        for idx in ['train','dev','test']:
            for i, tsuite in enumerate(sorted(glob.iglob(treebanks_path + idx + '/**'))):
                data[idx].append(self.process_one_testsuite(tsuite, idx, lextypes))
        return data


    def process_one_testsuite(self, tsuite, type, lextypes):
        ts = itsdb.TestSuite(tsuite)
        all_sentences = []
        parsed_sentences = []
        items = list(ts.processed_items())
        pairs = []
        total_tokens = 0
        print("{} sentences in corpus {} including possible sentences with no parse.".format(len(items), ts.path.stem))
        for response in items:
            all_sentences.append(response['i-input'])
            if len(response['results']) > 0:
                deriv = response.result(0).derivation()
                terminals = deriv.terminals()
                pairs.append(self.get_tokens_labels(terminals, lextypes))
        pc = ProcessedCorpus(ts.path.stem, type, pairs, all_sentences, parsed_sentences, total_tokens )
        return pc


    '''
    Input: a list of pydelphin itsdb terminals and a list of known lexical types (e.g. from the training data).
    Output: a list of tuples: the terminal orthographic form and its lexical type, if it is known, or <UNK> otherwise.
    '''
    def get_tokens_labels(self, terminals, lextypes):
        pairs = []
        for i,terminal, in enumerate(terminals):
            letype = str(lextypes.get(terminal.parent.entity, "<UNK>"))
            pairs.append((terminal.form, letype))
        return pairs



    def write_output(self, dest_path, data):
        for split_type in ['train', 'dev', 'test']:
            for ts_name in data[split_type]:
                with open(dest_path + ts_name, 'w') as f:
                    total = 0
                    for form, letype in data[split_type][ts_name]:
                        if not letype=='--EOS--':
                            str_pair = f'{form}\t{letype}'
                            f.write(str_pair + '\n')
                        else:
                            f.write('\n') # sentence separator
                            total += 1
                    print('Wrote {} sentences out.'.format(total))
