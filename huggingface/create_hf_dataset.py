################################################################
### Given pairs of tokens and tags, return a data structure  ###
### compatible with the HuggingFace Transformers library.    ###
###                                                          ###
################################################################

import sys
import json
from tempfile import NamedTemporaryFile
from datasets import Sequence, Value, ClassLabel, load_dataset, load_from_disk, Features, DatasetInfo
from transformers import AutoTokenizer

SPECIAL_TOKEN = -100
MWE = -100

LICENSE = "MIT LICENSE"

FLICKINGER2002 = """@article{flickinger2000building,
  title={On building a more efficient grammar by exploiting types},
  author={Flickinger, Dan},
  journal={Natural Language Engineering},
  volume={6},
  number={01},
  pages={15--28},
  year={2000},
  publisher={Cambridge University Press}
}"""
FLICKINGER2011 = """@incollection{Flickinger:11,
  author = {Flickinger, Dan},
  title = {Accuracy v. Robustness in Grammar Engineering},
  booktitle = {Language from a Cognitive Perspective: Grammar, Usage and Processing},
  editor = {Bender, Emily M. and Arnold, Jennifer E.},
  address = {Stanford, CA},
  publisher = {CSLI Publications},
  year = 2011,
  pages = {31--50}
}"""

def create_json(data, label_set, dtype):
    examples = []
    jsonf = NamedTemporaryFile("w", delete=False)
    total_sen = 0
    total_tok = 0
    skipped = 0
    min_word_len = 0 # if split == 'train' else 0
    with open(data, 'r') as f:
        sentences = [ sen for sen in f.read().split('\n\n') if sen != '' ]
        print("{} sentences read from the source dataset".format(len(sentences)))
        for sentence in sentences:
            sentence_tokens = []
            sentence_tags = []
            lines = sentence.split('\n')
            if len(lines) > min_word_len:
                for line in lines:
                    if line:
                        token, tag = line.split('\t')
                        sentence_tokens.append(token)
                        total_tok += 1
                        if dtype != 'test':
                            sentence_tags.append(tag)
                        else:
                            if tag in label_set:
                                sentence_tags.append(tag)
                            else:
                                sentence_tags.append('UNK')
                total_sen += 1
            else:
                #print('Skipped a sentence: {}'.format(sentence))
                skipped +=1
            if len(sentence_tokens) > 0:
                examples.append({"id": total_sen, "tokens": sentence_tokens, "tags": sentence_tags})
            else:
                print("No tokens in sentence {}".format(sentence))
    print("{} examples, {} tokens added to the {} dataset".format(total_sen, total_tok, dtype))
    print("Skipped {} sentences below minimum length {}.".format(skipped, min_word_len))

    with open(jsonf.name, 'w', encoding='utf8') as f:
        json.dump({'data': examples}, f, ensure_ascii=False)

    return jsonf.name


def create_json_files(data_files, label_set):
    train_list = []
    eval_list = []
    test_list = []

    train_json = NamedTemporaryFile("w", delete=False)
    eval_json = NamedTemporaryFile("w", delete=False)
    test_json = NamedTemporaryFile("w", delete=False)

    for split in data_files:
        total_sen = 0
        total_tok = 0
        file = data_files[split]
        skipped = 0
        min_word_len = 0 # if split == 'train' else 0
        with open(file, 'r') as f:
            sentences = [ sen for sen in f.read().split('\n\n') if sen != '' ]
            print("{} sentences read from the source dataset".format(len(sentences)))
            for sentence in sentences:
                sentence_tokens = []
                sentence_tags = []
                lines = sentence.split('\n')
                if len(lines) > min_word_len:
                    for line in lines:
                        if line:
                            token, tag = line.split('\t')
                            sentence_tokens.append(token)
                            total_tok += 1
                            if split != 'test':
                                sentence_tags.append(tag)
                            else:
                                if tag in label_set:
                                    sentence_tags.append(tag)
                                else:
                                    sentence_tags.append('UNK')
                    total_sen += 1
                else:
                    #print('Skipped a sentence: {}'.format(sentence))
                    skipped +=1
                if len(sentence_tokens) > 0:
                    if split == "train":
                        train_list.append({"id": total_sen, "tokens": sentence_tokens, "tags": sentence_tags})
                    elif split == "validation":
                        eval_list.append({"id": total_sen, "tokens": sentence_tokens, "tags": sentence_tags})
                    else:
                        test_list.append({"id": total_sen, "tokens": sentence_tokens, "tags": sentence_tags})
                else:
                    print("No tokens in sentence {}".format(sentence))
        print("{} examples, {} tokens added to the {} dataset".format(total_sen, total_tok, split))
        print("Skipped {} sentences below minimum length {}.".format(skipped, min_word_len))
    train_dic = {"data": train_list}
    eval_dic = {"data": eval_list}
    test_dic = {"data": test_list}

    with open(train_json.name, 'w', encoding='utf8') as f:
        json.dump(train_dic, f, ensure_ascii=False)

    with open(eval_json.name, 'w', encoding='utf8') as f:
        json.dump(eval_dic, f, ensure_ascii=False)

    with open(test_json.name, 'w', encoding='utf8') as f:
        json.dump(test_dic, f, ensure_ascii=False)

    return {"train": train_json.name, "validation": eval_json.name, "test": test_json.name}

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = SPECIAL_TOKEN if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(SPECIAL_TOKEN)
        else:
            # Same word as previous token
            new_labels.append(MWE)

    return new_labels

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        charspan = tokenized_inputs.token_to_chars(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def create_dataset(data, label_file, ds_info, ds_type):
    with open(label_file, 'r') as f:
        class_names = [l.strip() for l in f.readlines()]
    data_json = create_json(data, class_names,ds_type)
    num_labels = len(class_names)
    print('Number of labels:{}'.format(num_labels))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset = load_dataset(
        'json',
        data_files={ds_type: data_json},
        cache_dir="./cache", #'/media/olga/kesha/BERT/cache',
        features=Features(
            {
                "id": Value("int32"),
                "tokens": Sequence(Value("string")),
                "tags": Sequence(ClassLabel(names=list(class_names), num_classes=num_labels))
            }
        ),
        field="data",
    )
    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset[ds_type].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    if ds_info:
        update_info(dataset[ds_type]._info, ds_info, ds_type)
    print("Created dataset with shape {}".format(dataset.shape))
    return dataset

def create_dataset2(data_dir, label_file, test_subdataset, ds_info):
    with open(label_file, 'r') as f:
        class_names = [l.strip() for l in f.readlines()]
    data_tsv = {
        "train": data_dir + 'train/train',
        "validation": data_dir + 'dev/dev',
        "test": data_dir + 'test/' + test_subdataset
    }
    data_json = create_json_files(data_tsv, class_names)
    num_labels = len(class_names)
    print('Number of labels:{}'.format(num_labels))
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    dataset = load_dataset(
        "json",
        cache_dir="./cache", #'/media/olga/kesha/BERT/cache',
        data_files=data_json,
        features=Features(
            {
                "id": Value("int32"),
                "tokens": Sequence(Value("string")),
                "tags": Sequence(ClassLabel(names=list(class_names), num_classes=num_labels))
            }
        ),
        field="data",
    )
    print('Dataset loaded.')
    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={"tokenizer": tokenizer},
    )
    for split in ['train', 'validation', 'test']:
        update_info(dataset[split]._info, ds_info, split)
    return dataset

def update_info(cur_info, update_info, split):
    cur_info.citation = update_info.citation
    cur_info.homepage = update_info.homepage
    cur_info.description = "The recommended {} portion of the ".format(split.upper()) + update_info.description
    cur_info.license = update_info.license


def create_class_names(le):
    class_names = list(set([str(v) for v in list(le.lextypes.values())]))
    class_names.append('None_label')
    class_names.append('UNK')


def create_id2label_mappings(class_names):
    label2id = {v: i for i, v in enumerate(class_names)}
    id2label = {i: v for i, v in enumerate(class_names)}
    with open('label2id.json', 'w') as f:
        json.dump(label2id, f)
    with open('id2label.json', 'w') as f:
        json.dump(id2label, f)
    return label2id, id2label


if __name__ == '__main__':
    data = sys.argv[1] # A text file with token-tag pairs, sentences separated by additional newline
    output_dir = sys.argv[2]
    ds_type = sys.argv[3]
    ds_info = DatasetInfo(citation=FLICKINGER2002 + ';' + FLICKINGER2011,
                          homepage="https://github.com/delph-in/docs/wiki/ErgTop",
                          description="collection of manually verified treebanked data parsed "
                                      "by the English Recource Grammar, 2020 version.",
                          license=LICENSE)
    hf_ds = create_dataset(data, 'label_names.txt', ds_info, ds_type)
    #hf_ds = create_dataset2(data, 'label_names.txt', 'test', ds_info)
    hf_ds[ds_type].save_to_disk(output_dir)
