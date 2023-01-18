################################################################
### Given pairs of tokens and tags, return a data structure  ###
### compatible with the HuggingFace Transformers library.    ###
###                                                          ###
################################################################

import sys
import json
import numpy as np
from tempfile import NamedTemporaryFile
from datasets import Sequence, Value, ClassLabel, load_from_disk, Features
import evaluate
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer

SPECIAL_TOKEN = -100



def compute_metrics(eval_preds):
    with open('label_names.txt', 'r') as f:
        label_names = [l.strip() for l in f.readlines()]
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != SPECIAL_TOKEN] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != SPECIAL_TOKEN]
        for prediction, label in zip(predictions, labels)
    ]
    errors = collect_errors(true_predictions, true_labels)
    print("{} errors out of {} classified tokens".format(len(errors), len(true_predictions)))
    with open('errors.txt', 'w') as f:
        for e in errors:
            f.write(str(e) + '\n')
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def collect_errors(predictions, true_labels):
    errors = []
    for i, p in enumerate(predictions):
        if p != true_labels[i]:
            errors.append((p, true_labels[i]))

    return errors
if __name__ == '__main__':
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    dataset = load_from_disk(dataset_path) #'/media/olga/kesha/BERT/erg/dataset/'
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    with open('id2label.json','r') as f:
        id2label = json.load(f)
    with open('label2id.json','r') as f:
        label2id = json.load(f)

    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=1295,
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir= output_path+'/checkpoints/', #"/media/olga/kesha/BERT/erg/3e-5/"
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        num_train_epochs= 60,
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
    trainer.train()
    trainer.save_model(output_path + '/saved/')
