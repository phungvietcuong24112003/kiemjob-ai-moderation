
import pandas as pd
import re
from pathlib import Path

# Đường dẫn dataset
BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "dataset" / "raw" / "recruitment_moderation_dataset.csv"

LABEL_COLUMNS = [
    "upfront_fee_request",
    "job_scam",
    "pyramid_invest",
    "toxic_content",
    "discrimination",
    "spam_ads"
]

EXPECTED_COLUMNS = [
    "id", "title", "description", "text_combined",
    *LABEL_COLUMNS,
    "label_status", "source_type", "source_url",
    "source_group", "collected_at", "notes"
]

print("=" * 65)
print("KIEMJOB - DATASET QUALITY CHECK")
print("=" * 65)

# 1. Đọc dữ liệu
if not FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy dataset: {FILE}")

df = pd.read_csv(FILE, encoding="utf-8-sig")

missing_required = set(EXPECTED_COLUMNS) - set(df.columns)

if missing_required:
    raise ValueError(
        f"Dataset thiếu các cột bắt buộc: {missing_required}"
    )

print(f"\n[1] TỔNG QUAN")
print("Tổng số dòng:", len(df))
print("Tổng số cột:", len(df.columns))

# 2. Kiểm tra cấu trúc
print("\n[2] KIỂM TRA CẤU TRÚC")

missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
extra_columns = set(df.columns) - set(EXPECTED_COLUMNS)

print("Thiếu cột:", missing_columns)
print("Cột thừa:", extra_columns)
print("Đúng thứ tự cột:", list(df.columns) == EXPECTED_COLUMNS)

# 3. Kiểm tra ID
print("\n[3] KIỂM TRA ID")

print("ID bị thiếu:", df["id"].isna().sum())
print("ID trùng:", df["id"].duplicated().sum())

# 4. Kiểm tra phân bố nhãn
print("\n[4] PHÂN BỐ NHÃN")

for col in LABEL_COLUMNS:
    values = df[col]

    print(f"\n{col.upper()}")
    print("Positive (1):", (values == 1).sum())
    print("Negative (0):", (values == 0).sum())
    print("NULL:", values.isna().sum())
    print("Giá trị không hợp lệ:",
          (~values.isin([0, 1]) & values.notna()).sum())

# 5. Kiểm tra trạng thái nhãn
print("\n[5] LABEL STATUS")

print(df["label_status"].value_counts(dropna=False))

has_null_label = df[LABEL_COLUMNS].isna().any(axis=1)

invalid_status = (
    (has_null_label & (df["label_status"] != "NEEDS_REVIEW")) |
    (~has_null_label & (df["label_status"] != "LABELED"))
)

print("Số dòng không nhất quán label_status:", invalid_status.sum())

# 6. Kiểm tra dữ liệu thiếu
print("\n[6] DỮ LIỆU THIẾU")

for col in [
    "title", "description", "text_combined",
    "source_type", "source_group"
]:
    print(f"{col}: {df[col].isna().sum()}")

# 7. Kiểm tra text_combined
print("\n[7] KIỂM TRA TEXT_COMBINED")

def normalize(text):
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()

expected_text = (
    df["title"].fillna("").astype(str).str.strip()
    + ". "
    + df["description"].fillna("").astype(str).str.strip()
).str.strip()

actual_text = df["text_combined"].fillna("").astype(str).str.strip()

mismatch = (
    expected_text.map(normalize) != actual_text.map(normalize)
)

print("Số dòng text_combined không khớp:", mismatch.sum())
# 8. Kiểm tra trùng lặp
print("\n[8] KIỂM TRA TRÙNG LẶP")

df["_normalized_text"] = (
    df["title"].fillna("").map(normalize)
    + " "
    + df["description"].fillna("").map(normalize)
).str.lower()

print("Trùng title + description:",
      df["_normalized_text"].duplicated().sum())

print("Trùng text_combined:",
      actual_text.map(normalize).duplicated().sum())

# 9. Kiểm tra metadata
print("\n[9] METADATA")

for col in ["source_type", "source_group"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head(20))

# 10. Kiểm tra các mẫu có nhiều nhãn vi phạm
print("\n[10] MULTI-LABEL")

df["_positive_labels"] = df[LABEL_COLUMNS].fillna(0).sum(axis=1)

print("Không có nhãn vi phạm:",
      (df["_positive_labels"] == 0).sum())

print("Có ít nhất 1 nhãn vi phạm:",
      (df["_positive_labels"] >= 1).sum())

print("Có từ 2 nhãn vi phạm trở lên:",
      (df["_positive_labels"] >= 2).sum())

# 11. Xuất các dòng cần xem xét
review_rows = df[
    mismatch | invalid_status | has_null_label
].copy()

review_rows.to_csv(
    "dataset_rows_need_review.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nĐã xuất dataset_rows_need_review.csv")
print("\nHOÀN THÀNH KIỂM TRA!")
