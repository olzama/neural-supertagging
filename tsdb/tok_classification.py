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
                parsed_sentences.append(response['i-input'])
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
            letype = str(lextypes.get(terminal.parent.entity, "None_label"))
            pairs.append((terminal.form, letype))
        return pairs



    def write_output_by_corpus(self, dest_path, data):
        # Training and dev data is lumped all together
        for split_type in ['train', 'dev']:
            with open(dest_path + split_type + '/' + split_type, 'w') as f:
                total_sen = 0
                total_tok = 0
                for pc in data[split_type]:
                    total_sen, total_tok = self.write_out_one_corpus(f, pc, total_sen, total_tok)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, split_type))
        # Test data is kept separately by corpus, to be able to look at accuracy with different domains
        for pc in data['test']:
            with open(dest_path + 'test' + '/' + pc.name, 'w') as f:
                total_sen = 0
                total_tok = 0
                total_sen, total_tok = self.write_out_one_corpus(f, pc, total_sen, total_tok)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, pc.name))

    def write_out_one_corpus(self, f, pc, total_sen, total_tok):
        for sentence in pc.processed_data:
            for form, letype in sentence:
                str_pair = f'{form}\t{letype}'
                f.write(str_pair + '\n')
                total_tok += 1
            f.write('\n')  # sentence separator
            total_sen += 1
        return total_sen, total_tok

    def write_output_by_split(self, dest_path, data):
        for split_type in ['train', 'dev', 'test']:
            with open(dest_path + split_type + '/' + split_type, 'w') as f:
                total_sen = 0
                total_tok = 0
                for pc in data[split_type]:
                    total_sen, total_tok = self.write_out_one_corpus(f, pc, total_sen, total_tok)
                print('Wrote {} sentences, {} tokens out for {}.'.format(total_sen, total_tok, split_type))
