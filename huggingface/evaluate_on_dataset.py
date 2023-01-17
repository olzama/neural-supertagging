import sys
from datasets import load_from_disk, load_dataset, Features, Value, Sequence, ClassLabel
import evaluate
from transformers import AutoModelForTokenClassification, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
import numpy as np
from torch import nn, max as torch_max, IntTensor
from create_hf_dataset import create_json_files, align_labels_with_tokens, tokenize_and_align_labels
from letype_extractor import LexTypeExtractor

SPECIAL_TOKEN = -100

def test_eval_on_sentence(best_model, input_text):
    #input_text = "The text on which I test"
    input_text_tokenized = tokenizer.encode(input_text,
                                            truncation=True,
                                            padding=True,
                                            return_tensors="pt")
    prediction = best_model(input_text_tokenized)
    prediction_logits = prediction[0]
    sigmoid = nn.Sigmoid()
    probs = sigmoid(prediction_logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    max_prob_indices = [t.item() for t in list(torch_max(probs, dim=1)[1])]
    # predictions[np.where(probs >= 0.5)] = 1
    # turn predicted id's into actual label names
    # predicted_labels = [best_model.id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
    predicted_labels = [best_model.config.id2label[idx] for idx in max_prob_indices]
    print(predicted_labels)

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

best_model = AutoModelForTokenClassification.from_pretrained("/media/olga/kesha/BERT/erg/best/")
tokenizer = AutoTokenizer.from_pretrained("/media/olga/kesha/BERT/erg/best/")
#tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=best_model,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

best_model.eval()

with open('label_names.txt', 'r') as f:
    label_names = [l.strip() for l in f.readlines()]

dataset = load_from_disk('/media/olga/kesha/BERT/erg/')
predictions = trainer.predict(dataset['validation'])
print(5)
#test_eval_on_sentence(best_model=best_model, input_text="predictive analytivs encompasses a variety of techniques from statistics.")
#prediction_probs = tf.nn.softmax(prediction_logits,axis=1).numpy()
#print(f'The prediction probs are: {prediction_probs}')
