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


if __name__ == "__main__":
    best_model = AutoModelForTokenClassification.from_pretrained("/media/olga/kesha/BERT/erg/debug/")
    tokenizer = AutoTokenizer.from_pretrained("/media/olga/kesha/BERT/erg/debug/")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataset = load_from_disk('/media/olga/kesha/BERT/erg/dataset/')

    training_args = TrainingArguments(
        disable_tqdm=True,
        do_train=False,
        do_eval=True,
        do_predict=False,
        output_dir="/media/olga/kesha/BERT/erg/debug/"
    )

    trainer = Trainer(
        model=best_model,
        args=training_args,
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.evaluate()
    print(5)

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
    #test_eval_on_sentence(best_model=best_model, input_text="predictive analytivs encompasses a variety of techniques from statistics.")
    #prediction_probs = tf.nn.softmax(prediction_logits,axis=1).numpy()
    #print(f'The prediction probs are: {prediction_probs}')
