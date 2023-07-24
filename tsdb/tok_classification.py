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
            #for tok in terminal.tokens:
            #    pairs.append((terminal.form, letype))
            pairs.append((terminal.form, letype))
        return pairs

    def get_output_for_one_corpus(self, pc, total_sen, total_tok):
        output = ''
        for sentence in pc.processed_data:
            for form, letype in sentence:
                str_pair = f'{form}\t{letype}'
                output += str_pair + '\n'
                total_tok += 1
            output += '\n'  # sentence separator
            total_sen += 1
        return output, total_sen, total_tok

    def write_output_by_corpus(self, dest_path, data):
        print('Writing output to {}'.format(dest_path))
        # Training and dev data is lumped all together
        for split_type in ['train', 'dev']:
            with open(dest_path + split_type + '/' + split_type, self.output_format) as f:
                total_sen = 0
                total_tok = 0
                for pc in data[split_type]:
                    output, total_sen, total_tok = self.get_output_for_one_corpus(pc, total_sen, total_tok)
                    f.write(output)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, split_type))
        # Test data is kept separately by corpus, to be able to look at accuracy with different domains
        for pc in data['test']:
            with open(dest_path + 'test' + '/' + pc.name, self.output_format) as f:
                total_sen = 0
                total_tok = 0
                output, total_sen, total_tok = self.get_output_for_one_corpus(pc, total_sen, total_tok)
                f.write(output)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, pc.name))

    def write_output_by_split(self, dest_path, data):
        print('Writing output to {}'.format(dest_path))
        for split_type in ['train', 'dev', 'test']:
            with open(dest_path + split_type + '/' + split_type, self.output_format) as f:
                total_sen = 0
                total_tok = 0
                for pc in data[split_type]:
                    output, total_sen, total_tok = self.get_output_for_one_corpus(pc, total_sen, total_tok)
                    f.write(output)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, split_type))
