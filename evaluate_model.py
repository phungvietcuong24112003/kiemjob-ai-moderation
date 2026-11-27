
import csv
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    accuracy_score
)

DATASET_PATH = "dataset/raw/recruitment_moderation_dataset.csv"

LABEL_COLUMNS = [
    "upfront_fee_request",
    "job_scam",
    "pyramid_invest",
    "toxic_content",
    "discrimination",
    "spam_ads"
]

documents = []
labels = []

with open(DATASET_PATH, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["label_status"] != "LABELED":
            continue

        documents.append(row["text_combined"].strip())

        labels.append([
            int(row[label]) for label in LABEL_COLUMNS
        ])

X = np.array(documents)
y = np.array(labels)

# Chia lại đúng tập dữ liệu như lúc huấn luyện
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Tải mô hình đã lưu
vectorizer = joblib.load(
    "models/tfidf_vectorizer.joblib"
)

model = joblib.load(
    "models/multilabel_model.joblib"
)

# Chuyển đổi dữ liệu test
X_test_tfidf = vectorizer.transform(X_test)

# Dự đoán
y_pred = model.predict(X_test_tfidf)

# Lấy xác suất dự đoán của từng nhãn
y_proba = model.predict_proba(X_test_tfidf)

print("\n===== KIỂM TRA XÁC SUẤT DỰ ĐOÁN =====")

for i, label in enumerate(LABEL_COLUMNS):
    print(f"\n{label}")
    print("Xác suất cao nhất:", round(float(y_proba[:, i].max()), 4))
    print("Xác suất trung bình:", round(float(y_proba[:, i].mean()), 4))

    for threshold in [0.5, 0.3, 0.2, 0.1]:
        count = int((y_proba[:, i] >= threshold).sum())
        print(f"Ngưỡng {threshold}: {count} dự đoán nhãn 1")

print("\n===== KIỂM TRA SỐ LƯỢNG NHÃN =====")

for i, label in enumerate(LABEL_COLUMNS):
    print(f"\n{label}")
    print("Số nhãn thực tế:", int(y_test[:, i].sum()))
    print("Số nhãn mô hình dự đoán:", int(y_pred[:, i].sum()))

# Báo cáo từng nhãn
print("===== CLASSIFICATION REPORT =====")

print(classification_report(
    y_test,
    y_pred,
    target_names=LABEL_COLUMNS,
    zero_division=0
))

# Hamming Loss
print("===== HAMMING LOSS =====")

print(hamming_loss(y_test, y_pred))

# Exact Match Accuracy
print("===== SUBSET ACCURACY =====")

print(accuracy_score(y_test, y_pred))
