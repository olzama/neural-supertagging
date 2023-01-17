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
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

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
        learning_rate=5e-6,
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

    metric = evaluate.load("seqeval")

    trainer.train()
    trainer.save_model(output_path + '/saved/')
