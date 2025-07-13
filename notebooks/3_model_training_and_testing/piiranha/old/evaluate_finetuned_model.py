#Das ganze hat als Jupyter Notebook leider nicht geklappt, deswegen als py
#Das File ertellt die Performance Werte vom Model

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score

# === 1. Lade Modell und Tokenizer ===
model_path = "../../../../archive/old_workbooks/piranha_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

# === 2. Lade Testdaten ===
dataset = load_dataset("json", data_files="../data/converted_piranha/eval_converted_piranha.jsonl", split="train")

# Erstelle eine eindeutige Liste aller Labels
label_list = sorted(list(set(label for example in dataset for label in example['labels'])))
label_to_id = {label: i for i, label in enumerate(label_list)}


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []

    for i, label_seq in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label_seq[word_idx]])
            else:
                # Wenn du das Labeling-Format IOB verwendest, kannst du hier z. B. das gleiche Label wiederholen:
                label_ids.append(label_to_id[label_seq[word_idx]])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# === 4. Evaluation mit Trainer ===
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === 5. Vorhersage & Auswertung ===
predictions, labels, _ = trainer.predict(tokenized_dataset)
preds = np.argmax(predictions, axis=2)

# Labels decodieren
true_labels = []
true_predictions = []

for pred, label, tokens in zip(preds, labels, tokenized_dataset["tokens"]):
    true_labels.append([])
    true_predictions.append([])
    for p, l in zip(pred, label):
        if l != -100:
            true_labels[-1].append(model.config.id2label[l])
            true_predictions[-1].append(model.config.id2label[p])

# === 6. Ausgabe der Metriken ===
print("📊 Evaluationsergebnisse auf echten E-Mails:")
print(f"Accuracy:  {accuracy_score(true_labels, true_predictions):.4f}")
print(f"Precision: {precision_score(true_labels, true_predictions):.4f}")
print(f"Recall:    {recall_score(true_labels, true_predictions):.4f}")
print(f"F1 Score:  {f1_score(true_labels, true_predictions):.4f}")
