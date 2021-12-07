from delphin import tdl, itsdb
lextypes = {}  # mapping of lexical entry IDs to types
for event, obj, lineno in tdl.iterparse('/Users/olzama/Research/ERG/trunk/lexicon.tdl'):
    if event == 'TypeDefinition':
        lextypes[obj.identifier] = obj.supertypes[0]  # assume exactly 1

ts = itsdb.TestSuite('/Users/olzama/Research/ERG/trunk/tsdb/gold/mrs')
for response in ts.processed_items():
    result = response.result(0)
    deriv = result.derivation()
    pairs = [(t.form, t._parent.entity) for t in deriv.terminals()]
    print(' '.join(f'{form}/{lextypes.get(entity, entity)}'
                   for form, entity in pairs))
    break  # for one example only