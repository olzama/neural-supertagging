import sys
from datetime import datetime
import pathlib
from tsdb.feature_vectors import Feature_Vec_Extractor


if __name__ == "__main__":
    lexicons_path = sys.argv[1]
    treebanks_path = sys.argv[2]
    dt_str = '-'.join(str(datetime.now()).split()).replace(':','.')
    out_dir = sys.argv[3] + '-' + dt_str + '/'
    for split in ['train', 'dev', 'test']:
        for output_style in ['separate/', 'full/']:
            pathlib.Path(out_dir + output_style + split + '/').mkdir(parents=True, exist_ok=False)
    fve = Feature_Vec_Extractor()
    lextypes = fve.parse_lexicons(lexicons_path)
    data = fve.process_testsuites(treebanks_path, lextypes)
    fve.write_output_by_corpus(out_dir + 'separate/',data)
    fve.write_output_by_split(out_dir + 'full/',data)
