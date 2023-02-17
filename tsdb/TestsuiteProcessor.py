import glob, pathlib
import os
import pickle
from distutils.dir_util import copy_tree
from abc import ABC, abstractmethod
from delphin import tdl, itsdb



DEV = ['ws212', 'ecpa']
TEST = ['cb', 'ecpr', 'jhk', 'jhu', 'tgk', 'tgu', 'psk', 'psu', 'rondane',
             'vm32', 'ws213', 'ws214', 'petet', 'wsj23']
IGNORE = ['ntucle', 'omw', 'wlb03', 'wnb03', 'redwoods', 'wescience']
NONTRAIN = DEV + TEST + IGNORE

'''
Put the tsdb treebank folders into subdirectories train, dev, and test, according to the recommended split.
'''

def create_treebank_dirs(dir_str_stack, current_path):
    if not dir_str_stack:
        pathlib.Path(current_path).mkdir(parents=True, exist_ok=True)
        return
    for item in dir_str_stack[0]:
        create_treebank_dirs(dir_str_stack[1:], current_path + '/' + item)

def organize_treebanks(treebanks_path, output_dir):
    for tb in glob.iglob(treebanks_path + '/**'):
        tb_name = os.path.basename(os.path.normpath(tb))
        if tb_name in TEST:
            copy_tree(tb, output_dir + '/test/' + tb_name + '/')
        elif tb_name in DEV:
            copy_tree(tb, output_dir + '/dev/'+ tb_name + '/')
        elif tb_name not in IGNORE:
            copy_tree(tb, output_dir + '/train/'+ tb_name + '/')
'''
A ProcessedCorpus contains partial information from a [incr tsdb()] database.
It should have the list of all sentences from the corpus in a text form;
a list of all sentences which had a correct parse; the total number of tokens
as per the tokenizer that was used; the corpus type (train, test, or dev),
a name as it appears in the TSDB database; and finally:
data can look different depending on what kind of processing the corpus went through.
For example, data may contain a list of token-tag pairs, to be used for a token classification task downstream.
'''
class ProcessedCorpus:
    def __init__(self, name, type, data, all_sentences, parsed_sentences, total_tokens):
        self.name = name
        self.processed_data = data # E.g. list of token-tag pairs, or feature vectors
        # Lists of strings representing original text sentences in the corpus, in the same order they appear in the corpus
        self.parsed_sentence_list = parsed_sentences # Sentences for which a correct parse was recorded
        self.full_sentence_list = all_sentences # All sentences, regardless of whether they were parsed correctly
        self.total_tokens = total_tokens
        self.type = type


'''
This class is for experiments which require the data to be in different formats.
For example, for neural classifiers, the data for token classification might be expected in simple text format
such as "token\ttag" with sentences separated by an empty line. For classic ML classifiers, the data will be 
feature vectors, etc.
Thus, every TestsuiteProcessor must:
1) Read in the data from the [incr tsdb()] testsuites, in pydelphin implementation.
2) Process it so that it is in the desired format (the specifics will vary).
3) Store the processed data according to the training-development-test split, in a dictionary.
4) Write out the data into files, in the required format (the specifics will vary):
   4.1) stored by corpus, e.g. test/cb, test/wsj23
   4.2) stored all together, e.g. test/all_test
    
'''
class TestsuiteProcessor(ABC):

    @abstractmethod
    def get_observations(self, terminals, lextypes):
        pass

    @property
    @abstractmethod
    def output_format(self):
        pass

    @abstractmethod
    def get_output_for_one_corpus(self, pc, total_sen, total_tok):
        pass

    @abstractmethod
    def write_output_by_corpus(self, dest_path, data):
        pass

    @abstractmethod
    def write_output_by_split(self, dest_path, data):
        pass

    def parse_lexicons(self,lexicons):
        lextypes = {}  # mapping of lexical entry IDs to types
        for lexicon in glob.iglob(lexicons+'**'):
            for event, obj, lineno in tdl.iterparse(lexicon):
                if event == 'TypeDefinition':
                    lextypes[obj.identifier] = obj.supertypes[0]  # assume exactly 1
        return lextypes

    '''
    Takes a path to treebanks, already separated into train, dev, and test folders.
    The treebanks are in [incr tsdb()] format.
    Outputs a dictionary of three ProcessedCorpus lists (train, dev, and test).
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
        observations = []
        total_tokens = 0
        print("{} sentences in corpus {} including possible sentences with no parse.".format(len(items), ts.path.stem))
        for response in items:
            all_sentences.append(response['i-input'])
            if len(response['results']) > 0:
                parsed_sentences.append(response['i-input'])
                deriv = response.result(0).derivation()
                terminals = deriv.terminals()
                observations.append(self.get_observations(terminals, lextypes))
        pc = ProcessedCorpus(ts.path.stem, type, observations, all_sentences, parsed_sentences, total_tokens )
        return pc


