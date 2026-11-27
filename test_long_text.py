
import joblib

vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
model = joblib.load("models/multilabel_model.joblib")

LABEL_COLUMNS = [
    "UPFRONT_FEE_REQUEST",
    "JOB_SCAM",
    "PYRAMID_INVEST",
    "TOXIC_CONTENT",
    "DISCRIMINATION",
    "SPAM_ADS"
]

# Đoạn có dấu hiệu yêu cầu đóng phí
fee_text = """
Công ty đang tuyển nhân viên nhập liệu làm việc tại nhà.

Không yêu cầu kinh nghiệm, được đào tạo từ đầu.
Thu nhập từ 10 đến 15 triệu đồng mỗi tháng.

Ứng viên cần chuyển khoản trước 500.000 đồng
phí hồ sơ và phí kích hoạt tài khoản trước khi được nhận việc.

Sau khi thanh toán, công ty sẽ gửi tài khoản làm việc
và hướng dẫn công việc.
"""

# Nội dung tuyển dụng MSB (rút gọn để thử nghiệm)
msb_text = """
Mô tả công việc:
Chăm sóc, phục vụ khách hàng.
Lập kế hoạch kinh doanh, phát triển khách hàng.
Tư vấn các sản phẩm tín dụng và ngân hàng.

Yêu cầu ứng viên:
Cao đẳng trở lên chuyên ngành Tài chính - Ngân hàng.
Có kinh nghiệm bán hàng.
Có khả năng đàm phán, thuyết phục.

Thu nhập:
Thỏa thuận theo KPI.
Lương cứng không phụ thuộc doanh số.

Quyền lợi:
Được đào tạo và phát triển.
Có bảo hiểm và các chế độ phúc lợi.
Môi trường làm việc chuyên nghiệp.
"""

tests = {
    "A - Chi co noi dung thu phi": fee_text,
    "B - MSB + noi dung thu phi": msb_text + "\n" + fee_text,
    "C - Chi co noi dung MSB": msb_text
}

for name, text in tests.items():
    X = vectorizer.transform([text])

    predictions = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for i, label in enumerate(LABEL_COLUMNS):
        print(
            f"{label}: "
            f"{predictions[i]} "
            f"({probabilities[i]:.4f})"
        )
