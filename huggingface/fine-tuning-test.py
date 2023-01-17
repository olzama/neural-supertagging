from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, DataCollatorWithPadding
from huggingface_hub import notebook_login
from datasets import load_dataset
import numpy as np
import evaluate
#print(pipeline('sentiment-analysis')('we love you'))

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

wnut = load_dataset("wnut_17", cache_dir="/media/olga/kesha/BERT/cache")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)

# raw_datasets = load_dataset("glue", "mrpc", cache_dir="/media/olga/kesha/BERT/cache")
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# def tokenize_function(example):
#     return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
#
# tokenized_datasets = raw_datasets.map(tokenize_function,batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
# training_args = TrainingArguments(output_dir="/media/olga/kesha/BERT/test_trainer2")
#
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
#
# trainer = Trainer(model=model,
#                   args=training_args,
#                   train_dataset=tokenized_datasets["train"],
#                   eval_dataset=tokenized_datasets["validation"],
#                   data_collator=data_collator,
#                   tokenizer=tokenizer)
# trainer.train()
#
# predictions = trainer.predict(tokenized_datasets["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape)
#
# preds = np.argmax(predictions.predictions, axis=-1)
#
# metric = evaluate.load("glue", "mrpc")
# print(metric.compute(predictions=preds, references=predictions.label_ids))

# dataset = load_dataset("yelp_review_full", cache_dir='/media/olga/kesha/BERT/cache')
# print(dataset["train"][100])
# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
# tokenized_datasets = dataset.map(tokenize_function,batched=True)
#
# small_training_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
# training_args = TrainingArguments(output_dir="/media/olga/kesha/BERT/test_trainer", evaluation_strategy="epoch")
#
# metric = evaluate.load("accuracy")
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
# trainer = Trainer(model=model,
#                   args=training_args,
#                   train_dataset=small_training_dataset,
#                   eval_dataset=small_eval_dataset,
#                   compute_metrics=compute_metrics)
#
# trainer.train()