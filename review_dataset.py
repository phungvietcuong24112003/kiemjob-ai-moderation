
import csv
import random

FILE_PATH = "dataset/raw/recruitment_moderation_dataset.csv"

LABEL_COLUMNS = [
    "upfront_fee_request",
    "job_scam",
    "pyramid_invest",
    "toxic_content",
    "discrimination",
    "spam_ads"
]

# Đọc dataset
with open(FILE_PATH, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Chỉ lấy mẫu đã gán nhãn
labeled_rows = [
    row for row in rows
    if row["label_status"].strip() == "LABELED"
]

positive_rows = [
    row for row in labeled_rows
    if any(row[label] == "1" for label in LABEL_COLUMNS)
]

selected_rows = random.sample(
    positive_rows,
    min(30, len(positive_rows))
)

# In kết quả
for index, row in enumerate(selected_rows, start=1):
    print("\n" + "=" * 70)
    print(f"MẪU KIỂM TRA SỐ {index}")
    print("=" * 70)

    print("ID:", row["id"])
    print("Tiêu đề:", row["title"])
    print("Mô tả:", row["description"])

    print("\n--- NHÃN ---")

    for label in LABEL_COLUMNS:
        print(f"{label}: {row[label]}")

print("\nĐã hiển thị", len(selected_rows), "mẫu.")
