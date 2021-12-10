import unittest
import letype_extractor

LEXICONS = '/Users/olzama/Research/delphin/neural-supertagging/sample-data/toy-lexicons/'
MRS = '/Users/olzama/Research/delphin/neural-supertagging/sample-data/toy-testsuite/'

class ExampleTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


# Check that the Lecixal Type Extractor is counting things correctly,
# using the well-known MRS testsuite.
class MRSTest(unittest.TestCase):
    def test_mrs_testsuite(self):
        le = letype_extractor.LexTypeExtractor()
        le.parse_lexicons(LEXICONS)
        le.process_testsuites(MRS,le.lextypes)
        self.assertEqual(le.stats['corpora'][0]['items'],107) # should contain 107 sentences
        self.assertEqual(le.stats['corpora'][0]['noparse'], 0) # shoud parse all sentences
        self.assertEqual(len(le.stats['tokens']), 123) # shoud have 123 unique tokens
        self.assertEqual(sum(le.stats['tokens'].values()), 591) # should have 591 tokens total

if __name__ == '__main__':
    unittest.main()
