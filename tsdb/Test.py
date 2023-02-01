import unittest
from tsdb.tok_classification import Token_Tag_Extractor

LEXICONS = './test-treebanks/lexicons/'
TREEBANKS = './test-treebanks/treebanks-small/'

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
        


if __name__ == '__main__':
    unittest.main()
