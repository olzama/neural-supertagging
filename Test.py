import unittest
import pickle
from tsdb.tok_classification import Token_Tag_Extractor
from tsdb.feature_vectors import Feature_Vec_Extractor
from tsdb.feature_vectors_autoreg import Feature_Vec_Autoreg
from huggingface import create_hf_dataset
from classic_supertagging import vectorizer


LEXICONS = 'tsdb/test-treebanks/lexicons/'
TREEBANKS = 'tsdb/test-treebanks/treebanks-small/'
PROCESSED_TRAIN = 'tsdb/test-treebanks/processed-toktag-small/full/train/train'
PROCESSED_DEV = 'tsdb/test-treebanks/processed-toktag-small/full/dev/dev'
PROCESSED_TEST = 'tsdb/test-treebanks/processed-toktag-small/full/test/test'
LABELS = 'huggingface/label_names.txt'
TEST_DEST = 'tsdb/test-treebanks/test-tsdb-processing/'


# class ExampleTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, True)
#
#
# # Check that the Lexical Type Extractor is counting things correctly,
# # using the well-known MRS testsuite.
# class TestTTEProcessing(unittest.TestCase):
#     def test(self):
#         tte = Token_Tag_Extractor()  # This extracts token-tag pairs, per corpus, sentences separated by special character
#         lextypes = tte.parse_lexicons(LEXICONS)
#         data = tte.process_testsuites(TREEBANKS, lextypes)
#         self.assertEqual(len(data['train'][0].full_sentence_list),107) # should contain 107 sentences
#         self.assertEqual(len(data['train'][0].parsed_sentence_list), 107) # shoud parse all sentences
#         self.assertEqual(len(data['train'][0].processed_data), 107)  # should contain 107 token-tag pairs
#
# class TestCreateHFDataset(unittest.TestCase):
#     def test(self):
#         ds_train = create_hf_dataset.create_dataset(PROCESSED_TRAIN, LABELS, '', 'train')
#         ds_test = create_hf_dataset.create_dataset(PROCESSED_TEST, LABELS, '', 'test')
#         ds_dev = create_hf_dataset.create_dataset(PROCESSED_DEV, LABELS, '', 'validation')
#         self.assertEqual(ds_train['train'].shape[0], 132) #mrs + pest should have 132 sentences
#         self.assertEqual(ds_test.shape['test'][0], 125) # psk + tgk should have 125 sentences
#         self.assertEqual(ds_dev.shape['validation'][0],121) # ccs + esd should have 121 sentences
#
# class TestTSDB_to_HF(unittest.TestCase):
#     def test(self):
#         tte = Token_Tag_Extractor()  # This extracts token-tag pairs, per corpus, sentences separated by special character
#         lextypes = tte.parse_lexicons(LEXICONS)
#         data = tte.process_testsuites(TREEBANKS, lextypes)
#         tte.write_output_by_split(TEST_DEST, data)
#         ds_train = create_hf_dataset.create_dataset(TEST_DEST + '/train/train', LABELS, '', 'train')
#         self.assertEqual(ds_train['train'].shape[0], 132) #mrs + pest should have 132 sentences
#         #self.assertEqual(ds_test.shape['test'][0], 125) # psk + tgk should have 125 sentences
#         #self.assertEqual(ds_dev.shape['validation'][0],121) # ccs + esd should have 121 sentences
#
# class TestTSDB_to_FeatVecNonAutoReg(unittest.TestCase):
#     def test(self):
#         fve = Feature_Vec_Extractor()
#         lextypes = fve.parse_lexicons(LEXICONS)
#         data = fve.process_testsuites(TREEBANKS, lextypes)
#         self.assertEqual(len(data['train'][0].full_sentence_list),107) # should contain 107 sentences
#         self.assertEqual(len(data['train'][0].parsed_sentence_list), 107) # shoud parse all sentences
#         self.assertEqual(len(data['train'][0].processed_data), 107)  # should contain 107 token-tag pairs
#
# class TestTSDB_to_SCIKIT(unittest.TestCase):
#     def test(self):
#         fve = Feature_Vec_Extractor()
#         lextypes = fve.parse_lexicons(LEXICONS)
#         data = fve.process_testsuites(TREEBANKS, lextypes)
#         fve.write_output_by_split(TEST_DEST, data)
#         with open(TEST_DEST + '/train/train', 'rb') as f:
#             data_reloaded = pickle.load(f)
#         self.assertEqual(len(data_reloaded['ft']), 132)
#         fve.write_output_by_corpus(TEST_DEST,data)
#         with open(TEST_DEST + '/train/train', 'rb') as f:
#             data_reloaded = pickle.load(f)
#         self.assertEqual(len(data_reloaded['ft']), 132) # The training portion is still full training data, even when written out "by-corpus"; by-corpus is only for test.
#         with open(TEST_DEST + '/test/tgk', 'rb') as f:
#             data_reloaded = pickle.load(f)
#         self.assertEqual(len(data_reloaded['ft']), 85) # The tgk corpus has 85 parsed sentences

class TestTSDB_to_SCIKIT_Autoreg(unittest.TestCase):
    def test(self):
        fve = Feature_Vec_Autoreg()
        lextypes = fve.parse_lexicons(LEXICONS)
        data = fve.process_testsuites(TREEBANKS, lextypes)
        fve.write_output_by_split(TEST_DEST, data)
        with open(TEST_DEST + '/train/train', 'rb') as f:
            data_reloaded = pickle.load(f)
        self.assertEqual(len(data_reloaded[3]['ft']), 12) # there should be 12 sentences of length 3 in the training set
        total_train = 0
        for slen in data_reloaded:
            total_train += len(data_reloaded[slen]['ft'])
        self.assertEqual(total_train, 132) # There should still be 132 sentences in the training set


if __name__ == '__main__':
    unittest.main()
