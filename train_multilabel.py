
import csv
import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# =====================================
# 1. CẤU HÌNH
# =====================================

DATASET_PATH = "dataset/raw/recruitment_moderation_dataset.csv"
MODEL_DIR = "models"

LABEL_COLUMNS = [
    "upfront_fee_request",
    "job_scam",
    "pyramid_invest",
    "toxic_content",
    "discrimination",
    "spam_ads"
]

# =====================================
# 2. ĐỌC DỮ LIỆU
# =====================================

documents = []
labels = []

with open(DATASET_PATH, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:

        # Bỏ qua mẫu chưa được gán nhãn
        if row["label_status"] != "LABELED":
            continue

        # Lấy nội dung bài tuyển dụng
        text = row["text_combined"].strip()

        # Lấy 6 nhãn
        label_values = [
            int(row[label]) for label in LABEL_COLUMNS
        ]

        documents.append(text)
        labels.append(label_values)

X = np.array(documents)
y = np.array(labels)

print("Đọc dữ liệu thành công!")
print("Tổng số mẫu huấn luyện:", len(X))
print("Số nhãn:", len(LABEL_COLUMNS))

# =====================================
# 3. CHIA TRAINING VÀ TEST
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nKết quả chia dữ liệu:")
print("Training:", len(X_train))
print("Test:", len(X_test))

# =====================================
# 4. CHUYỂN VĂN BẢN THÀNH TF-IDF
# =====================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nChuyển đổi TF-IDF thành công!")
print("Số đặc trưng:", X_train_tfidf.shape[1])

# =====================================
# 5. KHỞI TẠO MÔ HÌNH
# =====================================

model = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )
)
# =====================================
# 6. HUẤN LUYỆN
# =====================================

print("\nĐang huấn luyện mô hình...")

model.fit(X_train_tfidf, y_train)

print("Huấn luyện hoàn tất!")

# =====================================
# 7. DỰ ĐOÁN THỬ TRÊN TẬP TEST
# =====================================

predictions = model.predict(X_test_tfidf)

print("\nKích thước kết quả dự đoán:", predictions.shape)

# =====================================
# 8. LƯU MÔ HÌNH
# =====================================

os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(
    vectorizer,
    os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
)

joblib.dump(
    model,
    os.path.join(MODEL_DIR, "multilabel_model.joblib")
)

print("\nĐã lưu mô hình thành công!")
print("Thư mục lưu:", MODEL_DIR)
