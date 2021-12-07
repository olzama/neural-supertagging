from delphin import tdl, itsdb




def parse_lexicons(lexicon_files):
    lextypes = {}  # mapping of lexical entry IDs to types
    for lexicon in lexicon_files:
        for event, obj, lineno in tdl.iterparse(lexicon):
            if event == 'TypeDefinition':
                lextypes[obj.identifier] = obj.supertypes[0]  # assume exactly 1
    return lextypes

def process_testsuites(testsuites,lextypes):
    for testsuite in testsuites:
        ts = itsdb.TestSuite(testsuite)
        pairs = []
        for response in ts.processed_items():
            result = response.result(0)
            deriv = result.derivation()
            for t in deriv.terminals():
                pairs.append((t.form, t._parent.entity))
            break
        with open('./output/'+ts.path.stem+'.txt', 'w') as f:
            for form, entity in pairs:
                letype = lextypes.get(entity, None)
                str_pair = f'{form}\t{letype}'
                print(str_pair)
                f.write(str_pair+'\n')

if __name__ == "__main__":
    lextypes = parse_lexicons(['/Users/olzama/Research/ERG/2020/lexicon.tdl','/Users/olzama/Research/ERG/2020/ple.tdl',
                               '/Users/olzama/Research/ERG/2020/gle.tdl','/Users/olzama/Research/ERG/2020/lexicon-rbst.tdl'])
    process_testsuites(['/Users/olzama/Research/ERG/2020/tsdb/gold/bcs'], lextypes)