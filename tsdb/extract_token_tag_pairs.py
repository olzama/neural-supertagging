import sys
from datetime import datetime
import pathlib
from tsdb.tok_classification import Token_Tag_Extractor


if __name__ == "__main__":
    lexicons_path = sys.argv[1]
    treebanks_path = sys.argv[2]
    dt_str = '-'.join(str(datetime.now()).split()).replace(':','.')
    run_id = sys.argv[3] + dt_str
    out_dir = './output/' + run_id + '/'
    for split in ['train', 'dev', 'test']:
        for output_style in ['separate/', 'full/']:
            pathlib.Path(out_dir + output_style + split + '/').mkdir(parents=True, exist_ok=False)
    tte = Token_Tag_Extractor() # This extracts token-tag pairs, per corpus, sentences separated by special character
    lextypes = tte.parse_lexicons(lexicons_path)
    data = tte.process_testsuites(treebanks_path, lextypes)
    tte.write_output_by_corpus(out_dir + 'separate/',data)
    tte.write_output_by_split(out_dir + 'full/',data)
