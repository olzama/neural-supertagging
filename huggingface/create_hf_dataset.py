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


def predict_test_set(input_test_seq, classifier, tokenizer):
    """
    Predict the syntaxis labels of a given test set in .seq format.
    """
    # Load test seq
    with open(input_test_seq, 'r') as f:
        sentences = f.read().split('\n\n')
        tokens_list = []
        pos_tags_list = []
        gold_labels_list = []

        i = 0
        for sentence in sentences:
            if len(sentence) > 0:
                # sys.stdout.write("\rPredicting sentence {}/{}".format(i, len(sentences)))
                i += 1
                sentence_tokens = []
                sentence_pos_tags = []
                sentence_gold_labels = []
                lines = sentence.split('\n')
                # Get tokens, pos tags and gold labels
                for line in lines:
                    if line:
                        token = line.split('\t')[0]
                        pos_tag = line.split('\t')[1]
                        gold_label = line.split('\t')[1]
                        sentence_tokens.append(token)
                        sentence_pos_tags.append(pos_tag)
                        sentence_gold_labels.append(gold_label)

                # Append to lists
                tokens_list.append(sentence_tokens)
                pos_tags_list.append(sentence_pos_tags)
                gold_labels_list.append(sentence_gold_labels)

    # Predict labels
    tokens_list = tokenizer(tokens_list, truncation=True, padding=True, is_split_into_words=True)
    predictions = classifier(tokens_list)

    print(predictions)
    print('\n')
    # print(f"Labeling accuracy: {sum(acc_list)/len(acc_list)}")

    # Generate output file
    output_test_seq = 'output_test.seq'
    with open(output_test_seq, 'w') as f:
        for i in range(len(tokens_list)):
            for j in range(len(tokens_list[i])):
                f.write(tokens_list[i][j] + '\t')
                f.write(pos_tags_list[i][j] + '\t')
                f.write('\n')
            f.write('\n\n')

def compute_metrics(eval_preds):
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != SPECIAL_TOKEN] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != SPECIAL_TOKEN]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def compute_metrics1(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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


# def create_hf_dicts(train_json, dev_json, test_json):
#     pass
#
# def serialize(files, file_names):
#     pass

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
    #metric = evaluate.load("seqeval")
    num_labels = len(class_names)

    label2id = {v: i for i, v in enumerate(class_names)}
    id2label = {i: v for i, v in enumerate(class_names)}

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

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

    dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={"tokenizer": tokenizer}

    )

    dataset.save_to_disk('/media/olga/kesha/BERT/erg/')


    #features = dataset['train'].features['labels']

    training_args = TrainingArguments(
        output_dir="/media/olga/kesha/BERT/erg/trainer/",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        num_train_epochs= 50,
        weight_decay=0.01,
        save_strategy = "no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    metric = evaluate.load("seqeval")

    trainer.train()


