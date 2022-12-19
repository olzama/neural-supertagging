################################################################
### Given pairs of tokens and tags, return a data structure  ###
### compatible with the HuggingFace Transformers library.    ###
###                                                          ###
################################################################

import sys
import json
from tempfile import NamedTemporaryFile
from datasets import Sequence, Value, ClassLabel, Features, load_dataset, Features
from letype_extractor import LexTypeExtractor

SPECIAL_TOKEN = -10000

def create_json_files(data_files):
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
                            sentence_tags.append(tag)

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


# def create_hf_dicts(train_json, dev_json, test_json):
#     pass
#
# def serialize(files, file_names):
#     pass

#if __name__ == 'main':
data_dir = sys.argv[1]
lexicons_dir = sys.argv[2]
print(1)
le = LexTypeExtractor()
le.parse_lexicons(lexicons_dir)
class_names = set([str(v) for v in list(le.lextypes.values())])
data_tsv = {
    "train": data_dir + 'train',
    "validation": data_dir + 'dev',
    "test": data_dir + 'test'
}
data_json = create_json_files(data_tsv)
metric = evaluate.load("seqeval")
num_labels = len(class_names["syntax_labels"])

print(5)
