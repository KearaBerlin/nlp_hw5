from datasets import load_dataset
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,\
    TrainingArguments, Trainer, pipeline, DataCollatorWithPadding, set_seed,\
    RobertaForSequenceClassification
import csv
dataset = load_dataset("Yaxin/SemEval2020Task9CodeSwitch")
test_data = dataset['test']
id2label = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
label2id = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}
model = RobertaForSequenceClassification.from_pretrained("roberta-base",
                                                         num_labels=3,
                                                         id2label=id2label,
                                                         label2id=label2id)
model.load_state_dict(torch.load("HW1Model.pth", map_location=torch.device('cpu')))
roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")


def preprocess_function(examples):
    listofstrings = [" ".join(tokens) for tokens in examples["tokens"]]
    return roberta_tokenizer(listofstrings, truncation=True)


tokenized_test = test_data.map(preprocess_function, batched=True)
raw_test = [" ".join(tokens) for tokens in test_data["tokens"]]
# raw_test_mini = raw_test[:10]
labels_test = test_data['label']
# print(labels_test)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=roberta_tokenizer, device=torch.device('cpu'))
start = time.time()
output = classifier(raw_test)

print(f'Inference time: {(time.time()-start)/60:.3f} minutes')
predictions = [label2id[entry['label']] for entry in output]
confidences = [entry['score'] for entry in output]
# print(predictions)
# print(labels_test[:10])
# print(confidences)

with open('hw5_output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sentence', 'True Label', 'Predicted Label', 'Confidence', 'Input Length', 'Token Length'])

tokenized_mini = roberta_tokenizer(raw_test, truncation=True)
for (prediction, label, confidence, sentence, token) in zip(predictions, labels_test, confidences, raw_test, tokenized_mini['input_ids']):
    sentence_length = len(sentence)
    token_length = len(token)
    with open('hw5_output.csv', 'a', newline='', encoding='UTF-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([sentence, label, prediction, confidence, sentence_length, token_length])