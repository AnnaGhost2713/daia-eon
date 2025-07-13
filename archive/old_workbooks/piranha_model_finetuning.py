#Hat leider als jpynb nicht geklappt, deswegen als py
#Das File finetuned das Model auf Basis der in piranha_finetuning_preparation erstellten Files

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Finetuning eines Token Classification Models mit Hugging Face Trainer

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from sklearn.metrics import classification_report
import numpy as np
import torch
import evaluate
import json
from pathlib import Path

import transformers
print("Transformers version:", transformers.__version__)


# Label-Liste gem. vorheriger Definition
GROUPED_LABELS = ["NAME", "ADRESSE", "VERTRAG", "ZAHLUNG", "TECHNISCHE_DATEN", "KONTAKT", "FIRMA", "DATUM"]
label_list = ["O"] + [f"{prefix}-{label}" for label in GROUPED_LABELS for prefix in ("B", "I")]
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

# Lade konvertierte Daten
train_data = []
base_path = Path(__file__).resolve().parent.parent / "data" / "converted_piranha"

with open(base_path / "train_converted_piranha.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        train_data.append(json.loads(line))

val_data = []
with open(base_path / "eval_converted_piranha.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        val_data.append(json.loads(line))

# In Hugging Face Dataset-Objekte konvertieren
train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(val_data)

# Tokenizer und Modell laden
tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
model = AutoModelForTokenClassification.from_pretrained("bert-base-german-cased", num_labels=len(label_list), id2label=id_to_label, label2id=label_to_id)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(label_to_id[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenisieren
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# Trainingsargumente
training_args = TrainingArguments(
    output_dir="./piranha_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Metriken
dataset_metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_predictions = [[id_to_label[p] for (p, l) in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    results = dataset_metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Training starten
trainer.train()

# Modell speichern
model.save_pretrained("./piranha_model")
tokenizer.save_pretrained("./piranha_model")

