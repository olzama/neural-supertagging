from delphin import tdl, itsdb
import glob, sys



def parse_lexicons(lexicons):
    lextypes = {}  # mapping of lexical entry IDs to types
    for lexicon in glob.iglob(lexicons+'**'):
        for event, obj, lineno in tdl.iterparse(lexicon):
            if event == 'TypeDefinition':
                lextypes[obj.identifier] = obj.supertypes[0]  # assume exactly 1
    return lextypes

def process_testsuites(testsuites,lextypes,stats):
    with open('./log.txt', 'w') as logf:
        for j,testsuite in enumerate(glob.iglob(testsuites+'**')):
            try:
                ts = itsdb.TestSuite(testsuite)
            except:
                print("ERROR: " + ts.path.stem)
                stats['failed corpora'].append({'name':ts.path.stem})
                logf.write("TESTSUITE ERROR: " + ts.path.stem + '\n')
            print("Processing " + ts.path.stem)
            stats['corpora'].append({'name':ts.path.stem})
            pairs = []
            items = list(ts.processed_items())
            stats['corpora'][j]['items'] = len(items)
            stats['corpora'][j]['noparse'] = 0
            for i,response in enumerate(items):
                if len(response['results']) > 0:
                    if i%100==0:
                        print("Processing item {} out of {}...".format(i,len(items)))
                    result = response.result(0)
                    deriv = result.derivation()
                    for t in deriv.terminals():
                        if not t.form in stats['tokens']:
                            stats['tokens'][t.form] = 0
                        stats['tokens'][t.form] += 1
                        pairs.append((t.form, t.parent.entity))
                else:
                    stats['corpora'][j]['noparse'] += 1
                    err = response['error'] if response['error'] else 'None'
                    logf.write(ts.path.stem + '\t' + str(response['i-id']) + '\t'
                               + response['i-input'] + '\t' + err +'\n')
            with open('./output/'+ts.path.stem+'.txt', 'w') as f:
                for form, entity in pairs:
                    letype = lextypes.get(entity, None)
                    str_pair = f'{form}\t{letype}'
                    f.write(str_pair+'\n')

if __name__ == "__main__":
    args = sys.argv[1:]
    stats = {'corpora': [], 'failed corpora': [], 'tokens': {}, 'total lextypes': 0}
    lextypes = parse_lexicons(args[0])
    stats['total lextypes'] = len(lextypes)
    process_testsuites(args[1],lextypes,stats)
    for c in stats['corpora']:
        for item in c:
            print(str(item) + ': ' + str(c[item]))
    print('Failed to load corpora:' + str(len (stats['failed corpora'])))
    for fc in stats['failed corpora']:
        print(fc)
    print('Total tokens in all corpora: ' + str(sum(stats['tokens'].values())))
    print('Total unique tokens: ' + str(len(stats['tokens'])))
    print('Total lextypes: ' + str(stats['total lextypes']))
