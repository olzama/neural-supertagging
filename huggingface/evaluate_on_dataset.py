import sys
from datasets import load_from_disk, load_dataset, Features, Value, Sequence, ClassLabel
import evaluate
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer
import numpy as np
from torch import nn, max as torch_max, IntTensor
from train_on_dataset import compute_metrics

SPECIAL_TOKEN = -100

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


def convert_predictions(predictions, model_config):
    txt_labels = []
    for p in list(predictions):
        txt_labels.append([model_config.id2label[idx] for idx in p])
    return txt_labels



if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    output_path = sys.argv[3]
    eval_type = sys.argv[4] if sys.argv[4]  == 'test' else 'validation'
    if len(sys.argv) == 6:
        test_sentence = sys.argv[5]
    best_model = AutoModelForTokenClassification.from_pretrained(model_path) #"/media/olga/kesha/BERT/erg/debug/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataset = load_from_disk(dataset_path)#'/media/olga/kesha/BERT/erg/dataset/')

    training_args = TrainingArguments(
        disable_tqdm=True,
        do_train=False,
        do_eval=False,
        do_predict=True,
        output_dir=output_path
    )

    trainer = Trainer(
        model=best_model,
        args=training_args,
        eval_dataset=dataset[eval_type],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    if len(sys.argv) == 5:
        #trainer.evaluate()
        predictions = trainer.predict(dataset['test'])
        #print(predictions)
        #print(predictions.label_ids)
        txt_predictions = convert_predictions(predictions.label_ids,best_model.config)
        print(txt_predictions)

    #best_model = AutoModelForTokenClassification.from_pretrained("/media/olga/kesha/BERT/erg/best/")
    #tokenizer = AutoTokenizer.from_pretrained("/media/olga/kesha/BERT/erg/best/")

    # trainer = Trainer(
    #     model=best_model,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    #     tokenizer=tokenizer,
    # )
    #
    # best_model.eval()
    #
    #
    # dataset = load_from_disk('/media/olga/kesha/BERT/erg/dataset/')
    # predictions = trainer.predict(dataset['validation'])
    # print(5)
    elif len(sys.argv) == 6:
        test_eval_on_sentence(best_model=best_model, input_text="predictive analytics encompasses a variety of techniques from statistics.")
    #prediction_probs = tf.nn.softmax(prediction_logits,axis=1).numpy()
    #print(f'The prediction probs are: {prediction_probs}')
