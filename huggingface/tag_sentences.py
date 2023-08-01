import sys
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import numpy as np
from torch import nn, max as torch_max
import torch
import re
from delphin import itsdb
from delphin.tokens import YYTokenLattice


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

def find_corresponding_toks(toks, terminal_span):
    tokens = []
    for tok in toks:
        if tok.lnk.data[0] == terminal_span[0] or tok.lnk.data[1] == terminal_span[1]:
            tokens.append(tok)
        if tok.lnk.data[1] > terminal_span[1]:
            return tokens
    return tokens

def extract_span(terminal):
    str_tok = terminal.tokens[0][1]
    from_match = re.search(r'\+FROM\s+\\"(\d+)\\"', str_tok)
    to_match = re.search(r'\+TO\s+\\"(\d+)\\"', str_tok)

    if from_match and to_match:
        from_value = int(from_match.group(1))
        to_value = int(to_match.group(1))
        return from_value, to_value
    else:
        return None


def extract_sentences(tsuite):
    sentences = []
    ts = itsdb.TestSuite(tsuite)
    items = list(ts.processed_items())
    for item in items:
        words = []
        terminals = None
        lattice = None
        char_spans = {}
        if len(item['results']) > 0:
            terminals = item.result(0).derivation().terminals()
        if (item['p-input']):
            lattice = YYTokenLattice.from_string(item['p-input'])
        if lattice and terminals:
            # toks_terms = find_corresponding_toks(lattice.tokens, terminals)
            for i, t in enumerate(terminals):
                terminal_span = extract_span(t)
                tokens = find_corresponding_toks(lattice.tokens, terminal_span)
                words.append(t.form)
                char_spans[str(terminal_span)] = []
                for tok in tokens:
                    char_spans[str(terminal_span)].append({'terminal-form': t.form, 'token-form': tok.form,
                                                              'start':tok.lnk.data[0],
                                                              'end': tok.lnk.data[1]})
        sentences.append({'sentence': item['i-input'], 'words':words, 'char_spans': char_spans})
    return sentences


def predict_tags_for_sentence(example, tokenizer, model, device):
    tokenized_input_with_mapping = tokenizer(example['sentence'], is_split_into_words=False, return_tensors="pt",
                                             return_offsets_mapping=True)
    pretokenized_input_with_mapping = tokenizer(example['words'], is_split_into_words=True, return_tensors="pt",
                                             return_offsets_mapping=True)
    tokenized_input = tokenizer(example['sentence'], is_split_into_words=False, return_tensors="pt")
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
    output = model(**tokenized_input)
    # Get character span for each token
    offset_mapping = tokenized_input_with_mapping['offset_mapping'][0].tolist()
    char_spans = [(start, end) for start, end in offset_mapping]
    word_ids = pretokenized_input_with_mapping.word_ids()
    # Get the predicted labels for each token
    predicted_labels = torch.argmax(output.logits, dim=2)[0].tolist()
    char_spans_to_tags = {char_span: tag for char_span, tag in zip(char_spans, predicted_labels)}
    adjusted = adjust_mapping(char_spans_to_tags, example['char_spans'])
    return adjusted

'''
Map BERT's tokenization to ACE's, relying on character offsets.
'''
def adjust_mapping(char_spans_to_tags, oracle_spans):
    adjusted = {}
    for char_span, tag in char_spans_to_tags.items():
        if str(char_span) in oracle_spans:
            adjusted[str(char_span)] = tag
        else:
            if char_span[0] != char_span[1]:
                multi_spans = find_multi_token_tag(oracle_spans, char_span)
                if multi_spans:
                    for ms in multi_spans:
                        adjusted[ms] = tag
    return adjusted

def find_multi_token_tag(oracle_spans, char_span):
    for oracle_span in oracle_spans:
        if oracle_spans[oracle_span][0]['start'] == char_span[0]:
            if len(oracle_spans[oracle_span]) == 1:
                return [str(oracle_span)]
            else:
                spans = []
                for span in oracle_spans[oracle_span]:
                    spans.append(str((span['start'], span['end'])))
                return spans
    return None

if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    best_model = AutoModelForTokenClassification.from_pretrained(model_path) #"/media/olga/kesha/BERT/erg/debug/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = best_model.to(device)
    sentences = extract_sentences(dataset_path)
    #example = 'The seeds produced as much as 20% corn.'
    all_predictions = []
    for example in sentences:
        char_spans_to_tags = predict_tags_for_sentence(example, tokenizer, model, device)
        all_predictions.append(char_spans_to_tags)
    with open(output_path + '/predictions.txt', 'w') as f:
        f.write(str(all_predictions))
        #f.write(json.dumps(char_spans_to_tags))
        # for char_span, tag in char_spans_to_tags.items():
        #     f.write(str(char_span) + '\t' + str(tag) + '\n')
