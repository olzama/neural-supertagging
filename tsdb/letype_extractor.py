from delphin import tdl, itsdb
import glob, sys, pathlib
from collections import OrderedDict
import pos_map
from datetime import datetime


DEV = ['ws212', 'ecpa']
TEST = ['cb', 'ecpr', 'jhk', 'jhu', 'tgk', 'tgu', 'psk', 'psu', 'rondane',
             'vm32', 'ws213', 'ws214', 'petet', 'wsj23']
IGNORE = ['ntucle', 'omw', 'wlb03', 'wnb03']
NONTRAIN = DEV + TEST + IGNORE


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
        self.processed_data = data # E.g. list of token-tag pairs
        # Lists of strings representing original text sentences in the corpus, in the same order they appear in the corpus
        self.parsed_sentence_list = parsed_sentences # Sentences for which a correct parse was recorded
        self.full_sentence_list = all_sentences # All sentences, regardless of whether they were parsed correctly
        self.total_tokens = total_tokens
        self.type = type


class LexTypeExtractor:
    def __init__(self, path_to_lexicons):
        self.stats = {'corpora': [], 'failed corpora': [], 'tokens': {}, 'total lextypes': 0}
        # Dictionary of processed corpora, with a train-dev-test split
        self.data = {'train': [], 'dev': [], 'test': []}
        # List of unique lexical types that occur in train and dev corpora, plus UNK
        self.lextypes = self.parse_lexicons(path_to_lexicons)

    def parse_lexicons(self,lexicons):
        lextypes = {}  # mapping of lexical entry IDs to types
        for lexicon in glob.iglob(lexicons+'**'):
            for event, obj, lineno in tdl.iterparse(lexicon):
                if event == 'TypeDefinition':
                    lextypes[obj.identifier] = obj.supertypes[0]  # assume exactly 1
        return lextypes
