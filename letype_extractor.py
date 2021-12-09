from delphin import tdl, itsdb
import glob, sys



def parse_lexicons(lexicons):
    lextypes = {}  # mapping of lexical entry IDs to types
    for lexicon in glob.iglob(lexicons+'**'):
        for event, obj, lineno in tdl.iterparse(lexicon):
            if event == 'TypeDefinition':
                lextypes[obj.identifier] = obj.supertypes[0]  # assume exactly 1
    return lextypes

def process_testsuites(testsuites,lextypes):
    log = ''
    with open('./errors.txt', 'w') as logf:
        for testsuite in glob.iglob(testsuites+'**'):
            try:
                ts = itsdb.TestSuite(testsuite)
                print("Processing " + ts.path.stem)
            except:
                print("ERROR: " + ts.path.stem)
                logf.write("TESTSUITE ERROR: " + ts.path.stem + '\n')
            pairs = []
            items = list(ts.processed_items())
            for i,response in enumerate(items):
                if len(response['results']) > 0:
                    if i%100==0:
                        print("Processing item {} out of {}...".format(i,len(items)))
                    result = response.result(0)
                    deriv = result.derivation()
                    for t in deriv.terminals():
                        pairs.append((t.form, t._parent.entity))
                else:
                    err = response['error'] if response['error'] else 'None'
                    log += ts.path.stem + '\t' + str(response['i-id']) + '\t' + response['i-input'] + '\t' + err +'\n'
                #break
            with open('./output/'+ts.path.stem+'.txt', 'w') as f:
                for form, entity in pairs:
                    letype = lextypes.get(entity, None)
                    str_pair = f'{form}\t{letype}'
                    f.write(str_pair+'\n')
            logf.write(log)


if __name__ == "__main__":
    args = sys.argv[1:]
    lextypes = parse_lexicons(args[0])
    process_testsuites(args[1],lextypes)