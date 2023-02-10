import sys

from TestsuiteProcessor import organize_treebanks

if __name__ == "__main__":
    dir_str_stack = [ ["full", "various-subsets"], ['tsdb'], ["train", "dev", "test"] ]
    organize_treebanks(dir_str_stack, sys.argv[1], sys.argv[2])

