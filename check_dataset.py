
import csv
from collections import Counter

FILE_PATH = "dataset/raw/recruitment_moderation_dataset.csv"

EXPECTED_COLUMNS = [
    "id", "title", "description", "text_combined",
    "upfront_fee_request", "job_scam", "pyramid_invest",
    "toxic_content", "discrimination", "spam_ads",
    "label_status", "source_type", "source_url",
    "source_group", "collected_at", "notes"
]

LABEL_COLUMNS = [
    "upfront_fee_request", "job_scam", "pyramid_invest",
    "toxic_content", "discrimination", "spam_ads"
]

errors = []
warnings = []

try:
    with open(FILE_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

except FileNotFoundError:
    print("Không tìm thấy file CSV!")
    print("Hãy kiểm tra lại FILE_PATH.")
    raise SystemExit

# 1. Kiểm tra cấu trúc
print("========== KIỂM TRA DATASET ==========")

if columns != EXPECTED_COLUMNS:
    errors.append("Tên hoặc thứ tự cột không đúng.")

    print("\nCột hiện tại:")
    print(columns)

    print("\nCột mong muốn:")
    print(EXPECTED_COLUMNS)
else:
    print("Cấu trúc 16 cột: OK")

# 2. Kiểm tra số lượng
print("\nTổng số mẫu:", len(rows))

if len(rows) != 600:
    warnings.append(f"Số mẫu là {len(rows)}, không phải 600.")

# 3. Kiểm tra ID
ids = [row.get("id", "").strip() for row in rows]

if len(ids) != len(set(ids)):
    errors.append("Có ID bị trùng.")

if any(not value for value in ids):
    errors.append("Có ID bị bỏ trống.")

# 4. Kiểm tra trạng thái và nhãn
status_counts = Counter()
label_counts = Counter()
source_counts = Counter()

for line, row in enumerate(rows, start=2):
    status = row.get("label_status", "").strip()
    source = row.get("source_type", "").strip()

    status_counts[status] += 1
    source_counts[source] += 1

    if source not in ("REAL", "SYNTHETIC"):
        errors.append(f"Dòng {line}: source_type không hợp lệ.")

    if status not in ("LABELED", "NEEDS_REVIEW"):
        errors.append(f"Dòng {line}: label_status không hợp lệ.")
        continue

    for label in LABEL_COLUMNS:
        value = row.get(label, "").strip()

        if status == "LABELED":
            if value not in ("0", "1"):
                errors.append(
                    f"Dòng {line}: {label} phải là 0 hoặc 1."
                )
            elif value == "1":
                label_counts[label] += 1

        elif status == "NEEDS_REVIEW":
            if value != "":
                errors.append(
                    f"Dòng {line}: NEEDS_REVIEW phải để trống nhãn."
                )

    # 5. Kiểm tra nội dung bắt buộc
    for field in ("title", "description", "text_combined"):
        if not row.get(field, "").strip():
            errors.append(f"Dòng {line}: thiếu {field}.")

    # Kiểm tra text_combined có chứa tiêu đề và mô tả
    title = row.get("title", "").strip()
    description = row.get("description", "").strip()
    combined = row.get("text_combined", "").strip()

    if title and description and (
        title not in combined or description not in combined
    ):
        warnings.append(
            f"Dòng {line}: text_combined có thể chưa đầy đủ."
        )

# 6. Kiểm tra trùng nội dung
seen = {}
duplicate_count = 0

for line, row in enumerate(rows, start=2):
    content = (
        row.get("title", "").strip().lower()
        + " "
        + row.get("description", "").strip().lower()
    )

    if content in seen:
        duplicate_count += 1
        warnings.append(
            f"Dòng {line}: trùng nội dung với dòng {seen[content]}."
        )
    else:
        seen[content] = line

# 7. In báo cáo
print("\n========== THỐNG KÊ ==========")
print("Trạng thái:", dict(status_counts))
print("Nguồn dữ liệu:", dict(source_counts))

print("\nSố mẫu có nhãn 1:")
for label in LABEL_COLUMNS:
    print(f"{label}: {label_counts[label]}")

print("\nSố nội dung trùng:", duplicate_count)

print("\n========== LỖI ==========")
if errors:
    for error in errors:
        print("[ERROR]", error)
else:
    print("Không phát hiện lỗi cấu trúc cơ bản.")

print("\n========== CẢNH BÁO ==========")
if warnings:
    for warning in warnings:
        print("[WARNING]", warning)
else:
    print("Không có cảnh báo.")

print("\n========== KẾT THÚC ==========")
