import csv

from sklearn.model_selection import train_test_split

# Đọc dữ liệu
documents = []
labels = []

with open("dataset/raw/practice_job_scam.csv", "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        text = row["title"] + " " + row["description"]

        documents.append(text)
        labels.append(int(row["job_scam"]))

# Chia dữ liệu: 80% Training, 20% Test
X_train, X_test, y_train, y_test = train_test_split(
    documents,
    labels,
    test_size=0.2,
    random_state=42
)

# Hiển thị kết quả
print("Tổng số bài:", len(documents))

print("Số bài Training:", len(X_train))
print("Số bài Test:", len(X_test))

print("Nhãn Training:", y_train)
print("Nhãn Test:", y_test)