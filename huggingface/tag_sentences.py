import glob
import sys, os
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
import numpy as np
from torch import nn, max as torch_max
import torch
import re
import json
from delphin import itsdb
from delphin.tokens import YYTokenLattice
from tsdb.TestsuiteProcessor import TestsuiteProcessor
from tsdb.tok_classification import Token_Tag_Extractor

SPECIAL_TOKEN = -100
MWE = 28996


def sentences_conllu(output_path):
    with open(output_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Extract sentences with word spans as tuples
    sentences_with_spans = []
    for sentence_data in data['sentences']:
        sentence_text = sentence_data.get('text',
                                          '')  # Assuming sentence text is stored as 'text' in each sentence data
        #word_spans = [{'word': word_info['word'], 'start': word_info['start'], 'end': word_info['end']}
                      #for word_info in sentence_data['words']]
        char_spans = [(word_info['start'], word_info['end']) for word_info in sentence_data['words']]
        sentences_with_spans.append({
            'sentence': sentence_text,
            'char_spans': char_spans
        })

    return sentences_with_spans

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

def extract_sentences_multidataset(profiles, lextypes):
    sentences = []
    for ppath in glob.iglob(profiles + '/**'):
        sentences.extend(extract_sentences(ppath, lextypes))
    return sentences

def extract_sentences(profiles, lextypes):
    sentences = []
    #for ppath in glob.iglob(profiles + '/**'):
    ts = itsdb.TestSuite(profiles)
    #ts = itsdb.TestSuite(ppath)
    items = list(ts.processed_items())
    for item in items:
        this_gold = []
        words = []
        terminals = None
        lattice = None
        char_spans = {}
        if len(item['results']) > 0:
            terminals = item.result(0).derivation().terminals()
        if (item['p-input']):
            lattice = YYTokenLattice.from_string(item['p-input'])
        if lattice and terminals:
            for i, t in enumerate(terminals):
                this_gold.append(str(lextypes.get(t.parent.entity, "None_label")))
                terminal_span = extract_span(t)
                tokens = find_corresponding_toks(lattice.tokens, terminal_span)
                words.append(t.form)
                char_spans[str(terminal_span)] = []
                for tok in tokens:
                    char_spans[str(terminal_span)].append({'terminal-form': t.form, 'token-form': tok.form,
                                                              'start':tok.lnk.data[0],
                                                              'end': tok.lnk.data[1]})
        assert len(words) == len(this_gold) == len(char_spans)
        sentences.append({'sentence': item['i-input'], 'words':words, 'char_spans': char_spans, 'gold': this_gold})
    return sentences


def predict_tags_for_sentence(example, tokenizer, model, device):
    tokenized_input_with_mapping = tokenizer(example['sentence'], is_split_into_words=False, return_tensors="pt",
                                             return_offsets_mapping=True)
    #pretokenized_input_with_mapping = tokenizer(example['words'], is_split_into_words=True, return_tensors="pt",
    #                                         return_offsets_mapping=True)
    tokenized_input = tokenizer(example['sentence'], is_split_into_words=False, return_tensors="pt")
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}
    #tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
    output = model(**tokenized_input)
    # Get character span for each token
    offset_mapping = tokenized_input_with_mapping['offset_mapping'][0].tolist()
    char_spans = [(start, end) for start, end in offset_mapping]
    #word_ids = pretokenized_input_with_mapping.word_ids()
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
    str_spans = [str(s) for s in oracle_spans]
    for char_span, tag in char_spans_to_tags.items():
        if str(char_span) in str_spans:
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
        if oracle_span[0] == char_span[0]:
            return [str(oracle_span)]
            # if len(oracle_spans[oracle_span]) == 1:
            #     return [str(oracle_span)]
            # else:
            #     spans = []
            #     for span in oracle_spans[oracle_span]:
            #         spans.append(str((span['start'], span['end'])))
            #     return spans
    return None

def convert_predictions(predictions, model_config):
    txt_labels = []
    for pred in predictions:
        txt_labels.append([model_config.id2label[idx] for idx in list(pred.values())])
    return txt_labels

def check_accuracy(predictions, sentences):
    correct = 0
    total = 0
    correct_excluding_gen_le = 0
    correct_sentences = 0
    correct_sen_excluding_gen_le = 0
    overpredicted = {}
    underpredicted = {}
    token_mistakes = {}
    for pred, sen in zip(predictions, sentences):
        whole_correct = True
        whole_correct_excluding_gen_le = True
        i=0
        for p, g in zip(pred, sen['gold']):
            if p == g:
                correct += 1
                correct_excluding_gen_le += 1
            else:
                whole_correct = False
                whole_correct_excluding_gen_le = False
                tok = sen['words'][i]
                if tok not in token_mistakes:
                    token_mistakes[tok] = 0
                token_mistakes[tok] += 1
                if p == "n_-_pn-gen_le":
                    whole_correct_excluding_gen_le = True
                    correct_excluding_gen_le += 1
                if p not in overpredicted:
                    overpredicted[p] = 0
                overpredicted[p] += 1
                if g not in underpredicted:
                    underpredicted[g] = 0
                underpredicted[g] += 1
                #print("Predicted: ", p, " Gold: ", g, "Sentence: ", sen['sentence'])
            total += 1
            i+=1
        if whole_correct:
            correct_sentences += 1
        if whole_correct_excluding_gen_le:
            correct_sen_excluding_gen_le += 1
    # Sort mistakes by frequency:
    overpredicted = {k: v for k, v in sorted(overpredicted.items(), key=lambda item: item[1], reverse=True)}
    print("Overpredicted top 10:")# print top 10 mistakes:
    for i, m in enumerate(overpredicted):
        if i < 3:
            print(m, overpredicted[m])
    underpredicted = {k: v for k, v in sorted(underpredicted.items(), key=lambda item: item[1], reverse=True)}
    # print top 10 mistakes:
    print("Underpredicted top 10:")
    for i, m in enumerate(underpredicted):
        if i < 10:
            print(m, underpredicted[m])
    token_mistakes = {k: v for k, v in sorted(token_mistakes.items(), key=lambda item: item[1], reverse=True)}
    for i, tok in enumerate(token_mistakes):
        if i < 10:
            print(tok, token_mistakes[tok])
    print("Total: ", total, " Correct: ", correct, " Accuracy: ", correct/total)
    return correct/total, correct_sentences/len(sentences), correct_excluding_gen_le/total, \
           correct_sen_excluding_gen_le/len(sentences)

def sentences_from_simple_file(dataset_path):
    sentences = []
    with open(dataset_path, 'r') as f:
        for line in f:
            sentences.append(line.strip())
    return sentences

if __name__ == "__main__":
    lexicons = sys.argv[1]
    model_path = sys.argv[2]
    dataset_path = sys.argv[3]
    output_path = sys.argv[4]
    # Create the directory if it does not exist:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    tte = Token_Tag_Extractor()
    lextypes = tte.parse_lexicons(lexicons)
    best_model = AutoModelForTokenClassification.from_pretrained(model_path) #"/media/olga/kesha/BERT/erg/debug/"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = best_model.to(device)
    #sentences = extract_sentences(dataset_path, lextypes)
    #sentences = extract_sentences_multidataset(dataset_path, lextypes)
    sentences = sentences_conllu(dataset_path)
    all_predictions = []
    for example in sentences:
        char_spans_to_tags = predict_tags_for_sentence(example, tokenizer, model, device)
        all_predictions.append(char_spans_to_tags)
    text_predictions = convert_predictions(all_predictions, model.config)
    #token_acc, sen_acc, token_acc_nogenle, sen_acc_nogenle = check_accuracy(text_predictions, sentences)
    #print("Token accuracy: {}, Sentence accuracy: {}".format(token_acc, sen_acc))
    #print("Token accuracy excluding generic le: {}, Sentence accuracy excluding generic_le: {}".format(token_acc_nogenle,
    #                                                                                                   sen_acc_nogenle))
    with open(output_path + '/sentences.txt', 'w') as f:
        for sent in sentences:
            f.write(sent['sentence'] + '\n')
    with open(output_path + '/spans.txt', 'w') as f:
        for sent in all_predictions:
            for i, span in enumerate(sent):
                f.write(str(span))
                if i < len(sent) - 1:
                    f.write('\t')
            f.write('\n')
    with open(output_path + '/predictions.txt', 'w') as f:
        for sent in all_predictions:
            for i,span in enumerate(sent):
                lbl = model.config.id2label[sent[span]]
                f.write(str(lbl))
                if i < len(sent) - 1:
                    f.write(', ')
            f.write('\n')
        #f.write(json.dumps(char_spans_to_tags))
        # for char_span, tag in char_spans_to_tags.items():
        #     f.write(str(char_span) + '\t' + str(tag) + '\n')
