import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Đọc dữ liệu
documents = []
labels = []

with open("dataset/raw/practice_job_scam.csv", "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        text = row["title"] + " " + row["description"]

        documents.append(text)
        labels.append(int(row["job_scam"]))

# Chuyển văn bản thành dữ liệu số
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Khởi tạo mô hình
model = LogisticRegression()

# Huấn luyện mô hình
model.fit(X, labels)

print("Huấn luyện mô hình thành công!")

# Bài tuyển dụng mới
new_job = [
    "Tuyển nhân viên xử lý đơn hàng online. Để được tham gia hệ thống, ứng viên cần thanh toán 200.000 VNĐ phí đăng ký và xác minh hồ sơ."
]

# Chuyển bài viết mới thành dữ liệu số
new_job_vector = vectorizer.transform(new_job)

# Dự đoán
prediction = model.predict(new_job_vector)

print("Kết quả dự đoán:", prediction[0])

if prediction[0] == 1:
    print("Cảnh báo: Bài tuyển dụng có dấu hiệu lừa đảo!")
else:
    print("Bài tuyển dụng chưa phát hiện dấu hiệu lừa đảo.")