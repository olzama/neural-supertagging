################################################################
### Given pairs of tokens and tags, return a data structure  ###
### compatible with the HuggingFace Transformers library.    ###
###                                                          ###
################################################################

import sys
import json
import numpy as np
from tempfile import NamedTemporaryFile
from datasets import Sequence, Value, ClassLabel, load_dataset, Features
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
from letype_extractor import LexTypeExtractor

SPECIAL_TOKEN = -100

def create_json_files(data_files, label_set):
    train_list = []
    eval_list = []
    test_list = []

    train_json = NamedTemporaryFile("w", delete=False)
    eval_json = NamedTemporaryFile("w", delete=False)
    test_json = NamedTemporaryFile("w", delete=False)

    for split in data_files:
        idx = 0
        file = data_files[split]
        with open(file, 'r') as f:
            sentences = f.read().split('\n\n')
            for sentence in sentences:
                idx += 1
                sentence_tokens = []
                sentence_tags = []

                lines = sentence.split('\n')
                if len(lines) > 1:
                    for line in lines:
                        if line:
                            token, tag = line.split('\t')
                            sentence_tokens.append(token)
                            if split != 'test':
                                sentence_tags.append(tag)
                            else:
                                if tag in label_set:
                                    sentence_tags.append(tag)
                                else:
                                    sentence_tags.append('UNK')
                if sentence_tokens != []:
                    if split == "train":
                        train_list.append({"id": idx, "tokens": sentence_tokens, "tags": sentence_tags})
                    elif split == "validation":
                        eval_list.append({"id": idx, "tokens": sentence_tokens, "tags": sentence_tags})
                    else:
                        test_list.append({"id": idx, "tokens": sentence_tokens, "tags": sentence_tags})

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
            new_labels.append(SPECIAL_TOKEN)

    return new_labels

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


if __name__ == '__main__':
    data_dir = sys.argv[1]
    lexicons_dir = sys.argv[2]
    le = LexTypeExtractor()
    le.parse_lexicons(lexicons_dir)
    class_names = list(set([str(v) for v in list(le.lextypes.values())]))
    class_names.append('None_label')
    class_names.append('UNK')
    data_tsv = {
        "train": data_dir + 'train',
        "validation": data_dir + 'dev',
        "test": data_dir + 'test'
    }
    data_json = create_json_files(data_tsv, class_names)
    num_labels = len(class_names)
    print('Number of labels:{}'.format(num_labels))

    label2id = {v: i for i, v in enumerate(class_names)}
    id2label = {i: v for i, v in enumerate(class_names)}

    with open('label2id.json', 'w') as f:
        json.dump(label2id,f)
    with open('id2label.json', 'w') as f:
        json.dump(id2label,f)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    dataset = load_dataset(
        "json",
        cache_dir='/media/olga/kesha/BERT/cache',
        data_files=data_json,
        features=Features(
            {
                "id": Value("int32"),
                "tokens": Sequence(Value("string")),
                "tags": Sequence(ClassLabel(names=list(class_names), num_classes=num_labels))
            }
        ),
        field="data"
    )

    print('Dataset loaded.')

    syntax_features = dataset['train'].features['tags']
    label_names = syntax_features.feature.names

    with open('label_names.txt', 'w') as f:
        for l in label_names:
            f.write(l+'\n')
    print('Saved label names in label_names.txt')

    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={"tokenizer": tokenizer}

    )

    dataset.save_to_disk('/media/olga/kesha/BERT/erg/dataset/')