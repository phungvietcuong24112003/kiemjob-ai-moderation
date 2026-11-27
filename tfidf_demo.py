from sklearn.feature_extraction.text import TfidfVectorizer

# 4 bài tuyển dụng mẫu
documents = [
    "Tuyển nhân viên kế toán lương 10 triệu",
    "Yêu cầu nạp tiền để mở khóa nhiệm vụ",
    "Tuyển nhân viên bán hàng lương 8 triệu",
    "Yêu cầu đặt cọc trước khi nhận việc"
]

# Khởi tạo công cụ TF-IDF
vectorizer = TfidfVectorizer()

# Chuyển văn bản thành dữ liệu số
tfidf_matrix = vectorizer.fit_transform(documents)

# Lấy danh sách từ vựng
words = vectorizer.get_feature_names_out()

print("Danh sách từ vựng:")
print(words)

print("\nKích thước dữ liệu:")
print(tfidf_matrix.shape)

print("\nMa trận TF-IDF:")
print(tfidf_matrix.toarray())