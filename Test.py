import unittest
from tsdb.tok_classification import Token_Tag_Extractor
from huggingface import create_hf_dataset

LEXICONS = 'tsdb/test-treebanks/lexicons/'
TREEBANKS = 'tsdb/test-treebanks/treebanks-small/'
PROCESSED_FULL = 'tsdb/test-treebanks/processed-toktag-small/full/'
LABELS = 'huggingface/label_names.txt'


class ExampleTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


# Check that the Lexical Type Extractor is counting things correctly,
# using the well-known MRS testsuite.
class MRSTest(unittest.TestCase):
    def test_tte_processing(self):
        tte = Token_Tag_Extractor()  # This extracts token-tag pairs, per corpus, sentences separated by special character
        lextypes = tte.parse_lexicons(LEXICONS)
        data = tte.process_testsuites(TREEBANKS, lextypes)
        self.assertEqual(len(data['train'][0].full_sentence_list),107) # should contain 107 sentences
        self.assertEqual(len(data['train'][0].parsed_sentence_list), 107) # shoud parse all sentences
        self.assertEqual(len(data['train'][0].processed_data), 107)  # should contain 107 token-tag pairs

class TestCreateHFDataset(unittest.TestCase):
    def test_create_hf_dataset(self):
        ds = create_hf_dataset.create_full_dataset(PROCESSED_FULL, LABELS)
        self.assertEqual(ds.shape['train'][0], 132) #mrs + pest should have 132 sentences
        self.assertEqual(ds.shape['test'][0], 125) # psk + tgk should have 125 sentences
        self.assertEqual(ds.shape['validation'][0],121) # ccs + esd should have 121 sentences

if __name__ == '__main__':
    unittest.main()
