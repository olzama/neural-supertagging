import sys
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
import numpy as np
from torch import nn, max as torch_max
from train_on_dataset import compute_metrics



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


def convert_predictions(predictions, model_config, vocab):
    txt_labels = []
    inputs = []
    token_nums = []
    prev = None
    for i,p in enumerate(list(predictions.label_ids)):
        this_sent_labels = []
        this_sent_tokens = []
        this_sent_token_nums = []
        prev_j = 0
        for j,idx in enumerate(p):
            #if idx == MWE:
            #    print("MWE")
            input_idx = predictions.inputs[i][j]
            if input_idx > 107: # meaningful words start at 107 in the vocab
                if idx != SPECIAL_TOKEN:
                    prev_j += 1
                    this_sent_labels.append(model_config.id2label[idx])
                    this_sent_token_nums.append(prev_j)
                    prev = idx
                else:
                    this_sent_labels.append(model_config.id2label[prev])
                    this_sent_token_nums.append(prev_j)
                this_sent_tokens.append(vocab[predictions.inputs[i][j]])

            #else:
            #    this_sent_labels.append('SPECIAL_TOKEN')
            #    this_sent_tokens.append('SPECIAL_TOKEN')
        txt_labels.append(this_sent_labels)
        inputs.append(this_sent_tokens)
        token_nums.append(this_sent_token_nums)
        #txt_labels.append([model_config.id2label[idx] for idx in p if idx != -100])
        #|inputs.append(vocab[idx] for idx in predictions.inputs[i]['input_ids'])
    return txt_labels, inputs, token_nums

def convert_inputs(inputs, tokenizer, vocab):
    txt_inputs = []
    spans = []
    for ii, i in enumerate(list(inputs)):
        sent = tokenizer.decode(i['input_ids'], skip_special_tokens=True)
        txt_inputs.append(sent)
        spans.append([])
        for t,tok in enumerate(i['input_ids']):
            print(vocab[tok])
            vtok = vocab[tok]
            lbl = i['labels'][t]
            if lbl == -100:
                print("SPECIAL_TOKEN")
            # Find vocab[tok]'s character span in sent:
            start, end = sent.find(vocab[tok]), sent.find(vocab[tok]) + len(vocab[tok])
            spans[ii].append((start, end, vocab[tok]))
            #print(super(PreTrainedTokenizerBase, tokenizer).tokens_to_chars(tok))
    return txt_inputs, spans


if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    best_model = AutoModelForTokenClassification.from_pretrained(model_path) #"/media/olga/kesha/BERT/erg/debug/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataset = load_from_disk(dataset_path)#'/media/olga/kesha/BERT/erg/dataset/')

    training_args = TrainingArguments(
        disable_tqdm=True,
        do_train=False,
        do_eval=False,
        do_predict=True,
        output_dir=output_path,
        eval_accumulation_steps=100,
        include_inputs_for_metrics=True
    )

    trainer = Trainer(
        model=best_model,
        args=training_args,
        #eval_dataset=dataset['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    # Read in the vocabulary:
    vocab = {}
    with open(model_path + '/vocab.txt', 'r') as f:
        vocab_lines = f.readlines()
    for i, w in enumerate(vocab_lines):
        vocab[i] = w.strip()
    # If the dataset was created from a single folder, then the whole thing gets passed to the predict()
    predictions = trainer.predict(dataset)
    # predictions = trainer.predict(dataset['test']) # If the dataset was created from separate train/dev/test folders
    txt_predictions, input_tokens, word_numbers = convert_predictions(predictions,best_model.config,vocab)
    txt_inputs, spans = convert_inputs(dataset,trainer.tokenizer,vocab)
    print(predictions.metrics)
    print('Saving predicted labels to ' + output_path + 'predictions.txt')
    with open(output_path + 'predictions.txt', 'w') as f:
        for p in txt_predictions:
            # strip the [ and ] from the list, strip single quotes from each item, and write to the file:
            f.write(str(p)[1:-1].replace("'", "") + '\n')
    # with open(output_path + 'word_nums.txt', 'w') as f:
    #     for p in word_numbers:
    #         f.write(str(p) + '\n')
    # with open(output_path + 'inputs.txt', 'w') as f:
    #     for p in input_tokens:
    #         f.write(str(p) + '\n')
    with open(output_path + 'spans.txt', 'w') as f:
        for p in spans:
            # Turn the list of the format: [(-1, 4, '[CLS]'), (0, 3, 'and')] into (-1,4) (0,3) and write to file such that there is no comma between (-1,4) and (0,3):
            for i,s in enumerate(p):
                if s[0] >= 0:
                    span = str(s[:2])
                    f.write(span)
                    if i < len(p)-1:
                        f.write('\t')
            f.write('\n')

