import sys
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
import numpy as np
from torch import nn, max as torch_max
from train_on_dataset import compute_metrics
import torch


SPECIAL_TOKEN = -100
MWE = 28996

def test_eval_on_sentence(best_model, input_text, tokenizer):
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


def predict_tags_for_sentence(example, tokenizer, model, device):
    tokenized_input_with_mapping = tokenizer(example, is_split_into_words=False, return_tensors="pt",
                                             return_offsets_mapping=True)
    tokenized_input = tokenizer(example, is_split_into_words=False, return_tensors="pt")
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
    output = model(**tokenized_input)
    # Get character span for each token
    offset_mapping = tokenized_input_with_mapping['offset_mapping'][0].tolist()
    char_spans = [(start, end) for start, end in offset_mapping]
    # Get the predicted labels for each token
    predicted_labels = torch.argmax(output.logits, dim=2)[0].tolist()
    char_spans_to_tags = {char_span: tag for char_span, tag in zip(char_spans, predicted_labels)}
    return char_spans_to_tags

if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    best_model = AutoModelForTokenClassification.from_pretrained(model_path) #"/media/olga/kesha/BERT/erg/debug/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = best_model.to(device)
    example = 'The seeds produced as much as 20% corn.'
    char_spans_to_tags = predict_tags_for_sentence(example, tokenizer, model, device)
    with open(output_path + '/predictions.txt', 'w') as f:
        #f.write(json.dumps(char_spans_to_tags))
        for char_span, tag in char_spans_to_tags.items():
            f.write(str(char_span) + '\t' + str(tag) + '\n')
