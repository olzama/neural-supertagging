import sys

from TestsuiteProcessor import organize_treebanks, create_treebank_dirs

if __name__ == "__main__":
    dir_str_stack = [ ["full", "various-subsets"], ['tsdb'], ["train", "dev", "test"] ]
    create_treebank_dirs(dir_str_stack, sys.argv[1])
    organize_treebanks(sys.argv[2],sys.argv[1] + '/full/tsdb/')

