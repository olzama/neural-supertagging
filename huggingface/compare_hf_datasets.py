import sys
import json
from datasets import Sequence, Value, ClassLabel, load_from_disk, Features
from transformers import AutoTokenizer


if __name__ == '__main__':
    with open('id2label.json','r') as f:
        id2label = json.load(f)
    with open('label2id.json','r') as f:
        label2id = json.load(f)

    with open('debug-datasets/vocab-reg.txt', 'r') as f:
        vocab = f.readlines()

    dataset1_path = sys.argv[1]
    dataset2_path = sys.argv[2]


    dataset_record = load_from_disk(dataset1_path)
    print('Loaded dataset which yields best numbers. Shape: {}'.format(dataset_record.shape))

    dataset_regular = load_from_disk(dataset2_path)
    print('Loaded dataset created with recent code. Shape: {}'.format(dataset_regular.shape))

    same = []
    in_record_but_not_in_regular = {'train': [], 'dev': [], 'test': [] }
    in_regular_but_not_in_record = {'train': [], 'dev': [], 'test': [] }

    #for split in ['train', 'validation', 'test']:
    for split in ['train']:
        items_record = dataset_record[split]
        all_items_record = {}
        all_items_reg = {}
        for i, item in enumerate(items_record):
            # look at the item number i in both datasets, converting to text
            if i < len(dataset_regular[split]):
                item_reg = dataset_regular[split][i]
                txt_item_record = ' '.join([vocab[wid] for wid in item['input_ids']])
                txt_item_reg = ' '.join([vocab[wid] for wid in item_reg['input_ids']])
                # add the counts for these items, how many times they occur in each dataset
                if not txt_item_record in all_items_record:
                    all_items_record[txt_item_record] = 0
                all_items_record[txt_item_record] += 1
                if not txt_item_reg in all_items_reg:
                    all_items_reg[txt_item_reg] = 0
                all_items_reg[txt_item_reg] +=1
                # If item number i is not the same in both datasets, report
                #if item_reg['input_ids'] != item['input_ids']:
                #    print('Mismatch in {} item {}!\nRecord dataset item: {}\nRegular dataset item:{}'.format(split, i, txt_item_record, txt_item_reg))
        print(5)


