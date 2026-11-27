
import csv

DATASET_PATH = "dataset/raw/recruitment_moderation_dataset.csv"

LABEL_COLUMNS = [
    "upfront_fee_request",
    "job_scam",
    "pyramid_invest",
    "toxic_content",
    "discrimination",
    "spam_ads"
]

suspicious_rows = []

with open(DATASET_PATH, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["label_status"] != "LABELED":
            continue

        content = (
            row["title"] + " " + row["description"]
        ).lower()

        # =====================================
        # 1. KIỂM TRA UPFRONT_FEE_REQUEST
        # =====================================

        fee_keywords = [
            "nạp tiền",
            "đặt cọc",
            "đóng phí",
            "phí hồ sơ",
            "phí đào tạo",
            "mua gói",
            "mua bộ sản phẩm",
            "mua thẻ thành viên",
            "phí kích hoạt",
            "chuyển khoản trước"
        ]

        negative_fee_phrases = [
            "không thu phí",
            "không thu bất kỳ khoản phí",
            "không yêu cầu nộp phí",
            "không yêu cầu đóng phí",
            "không cần nộp phí",
            "không cần đóng phí",
            "không thu phí đào tạo",
            "không thu phí nhận việc",
            "không cần mua gói",
            "không yêu cầu mua gói",
            "không cần đặt cọc",
            "không yêu cầu đặt cọc"
        ]

        if row["upfront_fee_request"] == "0":

            # Loại bỏ những cụm từ phủ định trước khi kiểm tra
            fee_content = content

            for phrase in negative_fee_phrases:
                fee_content = fee_content.replace(phrase, "")

            # Tìm từ khóa liên quan đến phí
            for keyword in fee_keywords:
                if keyword in fee_content:
                    suspicious_rows.append({
                        "id": row["id"],
                        "title": row["title"],
                        "description": row["description"],
                        "label": "UPFRONT_FEE_REQUEST",
                        "reason": (
                            f"Có từ khóa '{keyword}' nhưng nhãn "
                            "UPFRONT_FEE_REQUEST = 0"
                        )
                    })
                    break

        # =====================================
        # 2. KIỂM TRA PYRAMID_INVEST
        # =====================================

        
        pyramid_group_a = [
            "f1",
            "f2",
            "f3",
            "tuyến dưới",
            "tuyến trên",
            "tuyến thứ 2",
            "nhiều tầng"
        ]

        pyramid_group_b = [
            "mời thành viên",
            "tuyển người",
            "giới thiệu người",
            "tuyển đủ",
            "người được mời"
        ]

        pyramid_group_c = [
            "mua gói",
            "góp vốn",
            "cổ phần ảo",
            "tái đầu tư",
            "gói đầu tư",
            "vốn tham gia",
            "đầu tư sinh lời"
        ]

        pyramid_group_d = [
            "hoa hồng từ",
            "hưởng chiết khấu từ",
            "nhận phần trăm từ",
            "nhận thưởng từ người",
            "hoa hồng tuyến"
        ]

        if row["pyramid_invest"] == "0":

            has_group_a = any(
                keyword in content for keyword in pyramid_group_a
            )

            has_group_b = any(
                keyword in content for keyword in pyramid_group_b
            )

            has_group_c = any(
                keyword in content for keyword in pyramid_group_c
            )

            has_group_d = any(
                keyword in content for keyword in pyramid_group_d
            )

            # Cảnh báo khi có dấu hiệu từ ít nhất 2 nhóm
            matched_groups = sum([
                has_group_a,
                has_group_b,
                has_group_c,
                has_group_d
            ])

            if matched_groups >= 2:
                suspicious_rows.append({
                    "id": row["id"],
                    "title": row["title"],
                    "description": row["description"],
                    "label": "PYRAMID_INVEST",
                    "reason": (
                        "Có dấu hiệu từ nhiều nhóm đặc trưng "
                        "của mô hình đa cấp/đầu tư."
                    )
                })

            
        # =====================================
        # 3. KIỂM TRA DISCRIMINATION
        # =====================================

        discrimination_keywords = [
            "không nhận người miền",
            "không nhận người nghệ an",
            "không nhận người thanh hóa",
            "chỉ nhận nam",
            "chỉ nhận nữ",
            "không nhận nam",
            "không nhận nữ",
            "không nhận người đã có gia đình",
            "không nhận phụ nữ có con nhỏ",
            "ngoại hình xinh",
            "gương mặt đẹp",
            "cao trên 1m",
            "ưu tiên người trẻ đẹp"
        ]

        if row["discrimination"] == "0":

            for keyword in discrimination_keywords:
                if keyword in content:
                    suspicious_rows.append({
                        "id": row["id"],
                        "title": row["title"],
                        "description": row["description"],
                        "label": "DISCRIMINATION",
                        "reason": (
                            f"Có dấu hiệu phân biệt đối xử: '{keyword}'"
                        )
                    })
                    break




# =====================================
# 3. HIỂN THỊ KẾT QUẢ
# =====================================

print("TỔNG SỐ CẢNH BÁO:", len(suspicious_rows))

for item in suspicious_rows:
    print("------------------------------")
    print("ID:", item["id"])
    print("Tiêu đề:", item["title"])
    print("Nhãn cần kiểm tra:", item["label"])
    print("Lý do:", item["reason"])
    print("Mô tả:", item["description"])
