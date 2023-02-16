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

from tsdb.TestsuiteProcessor import ProcessedCorpus, TestsuiteProcessor

EOS_TOKEN = 'EOS'

class Token_Tag_Extractor(TestsuiteProcessor):

    def __init__(self):
        pass

    @property
    def output_format(self):
        return 'w'

    '''
    Input: a list of pydelphin itsdb terminals and a list of known lexical types (e.g. from the training data).
    Output: a list of tuples: the terminal orthographic form and its lexical type, if it is known, or <UNK> otherwise.
    '''
    def get_observations(self, terminals, lextypes):
        pairs = []
        for i,terminal, in enumerate(terminals):
            letype = str(lextypes.get(terminal.parent.entity, "None_label"))
            pairs.append((terminal.form, letype))
        return pairs

    def write_out_one_corpus(self, f, pc, total_sen, total_tok):
        for sentence in pc.processed_data:
            for form, letype in sentence:
                str_pair = f'{form}\t{letype}'
                f.write(str_pair + '\n')
                total_tok += 1
            f.write('\n')  # sentence separator
            total_sen += 1
        return total_sen, total_tok

