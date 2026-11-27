
import joblib

# Tải mô hình đã huấn luyện
vectorizer = joblib.load(
    "models/tfidf_vectorizer.joblib"
)

model = joblib.load(
    "models/multilabel_model.joblib"
)

LABEL_COLUMNS = [
    "UPFRONT_FEE_REQUEST",
    "JOB_SCAM",
    "PYRAMID_INVEST",
    "TOXIC_CONTENT",
    "DISCRIMINATION",
    "SPAM_ADS"
]

# Bài tuyển dụng cần kiểm tra
title = "Tuyển cộng tác viên nhập liệu tại nhà"

description = (
    "Công việc nhập liệu online, thu nhập 500.000 đồng/ngày. "
    "Ứng viên cần nạp 500.000 đồng để kích hoạt tài khoản "
    "trước khi nhận nhiệm vụ."
)

# Kết hợp tiêu đề và mô tả
text = f"[TIÊU ĐỀ] {title} [MÔ TẢ] {description}"

# Chuyển văn bản thành TF-IDF
text_tfidf = vectorizer.transform([text])

# Dự đoán 6 nhãn
prediction = model.predict(text_tfidf)[0]

# Lấy xác suất dự đoán
probabilities = model.predict_proba(text_tfidf)[0]

print("===== KẾT QUẢ KIỂM DUYỆT =====")
print("Tiêu đề:", title)

for i, label in enumerate(LABEL_COLUMNS):
    result = "CÓ DẤU HIỆU" if prediction[i] == 1 else "KHÔNG PHÁT HIỆN"

    print(
        f"{label}: {result} "
        f"(Xác suất: {probabilities[i]:.4f})"
    )
