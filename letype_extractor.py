from delphin import tdl, itsdb




def parse_lexicons(lexicon_files):
    lextypes = {}  # mapping of lexical entry IDs to types
    for lexicon in lexicon_files:
        for event, obj, lineno in tdl.iterparse(lexicon):
            if event == 'TypeDefinition':
                lextypes[obj.identifier] = obj.supertypes[0]  # assume exactly 1
    return lextypes

def process_testsuite(testsuite_path,lextypes):
    ts = itsdb.TestSuite(testsuite_path)
    for response in ts.processed_items():
        result = response.result(0)
        deriv = result.derivation()
        pairs = [(t.form, t._parent.entity) for t in deriv.terminals()]
        print(' '.join(f'{form}/{lextypes.get(entity, entity)}'
                       for form, entity in pairs))
        break  # for one example only


if __name__ == "__main__":
    lextypes = parse_lexicons(['/Users/olzama/Research/ERG/trunk/lexicon.tdl'])
    process_testsuite('/Users/olzama/Research/ERG/trunk/tsdb/gold/mrs', lextypes)