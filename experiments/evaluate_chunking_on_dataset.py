import csv
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, accuracy_score

sys.stdout.reconfigure(encoding='utf-8')

# Import chunk_text & predict_chunk_based from test_suite
from test_suite_10_cases import chunk_text, predict_chunk_based, predict_whole_doc, LABEL_COLUMNS

DATASET_PATH = "dataset/raw/recruitment_moderation_dataset.csv"

# Load dataset
documents = []
labels = []
titles = []
descriptions = []

with open(DATASET_PATH, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row["label_status"] != "LABELED":
            continue
        titles.append(row["title"].strip())
        descriptions.append(row["description"].strip())
        documents.append(row["text_combined"].strip())
        labels.append([int(row[l.lower()]) for l in LABEL_COLUMNS])

X_titles = np.array(titles)
X_descs = np.array(descriptions)
X_docs = np.array(documents)
y = np.array(labels)

indices = np.arange(len(y))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

y_test = y[test_idx]
titles_test = X_titles[test_idx]
descs_test = X_descs[test_idx]

print(f"Total test samples: {len(test_idx)}")

# 1. Whole-Doc evaluation
y_pred_old = []
for t, d in zip(titles_test, descs_test):
    res = predict_whole_doc(t, d)
    preds = [res[l]['prediction'] for l in LABEL_COLUMNS]
    y_pred_old.append(preds)
y_pred_old = np.array(y_pred_old)

# 2. Chunk-Based evaluation
y_pred_new = []
for t, d in zip(titles_test, descs_test):
    res = predict_chunk_based(t, d)
    preds = [res['results'][l]['prediction'] for l in LABEL_COLUMNS]
    y_pred_new.append(preds)
y_pred_new = np.array(y_pred_new)

print("\n" + "=" * 60)
print("1. CLASSIFICATION REPORT - PHƯƠNG PHÁP CŨ (WHOLE-DOC)")
print("=" * 60)
print(classification_report(y_test, y_pred_old, target_names=LABEL_COLUMNS, zero_division=0))
print("Hamming Loss:", hamming_loss(y_test, y_pred_old))
print("Subset Accuracy:", accuracy_score(y_test, y_pred_old))

print("\n" + "=" * 60)
print("2. CLASSIFICATION REPORT - PHƯƠNG PHÁP MỚI (CHUNK-BASED)")
print("=" * 60)
print(classification_report(y_test, y_pred_new, target_names=LABEL_COLUMNS, zero_division=0))
print("Hamming Loss:", hamming_loss(y_test, y_pred_new))
print("Subset Accuracy:", accuracy_score(y_test, y_pred_new))

