import joblib

from fastapi import FastAPI
from pydantic import BaseModel
from moderator import moderate_job_post, LABEL_COLUMNS

# =====================================
# 1. KHỞI TẠO API
# =====================================

app = FastAPI(
    title="KiemJob AI Moderation API",
    description="API kiểm duyệt nội dung tuyển dụng bằng AI (Chunk-based Classification)",
    version="2.0.0"
)

# =====================================
# 2. TẢI MÔ HÌNH
# =====================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer.joblib"
)

model = joblib.load(
    "models/multilabel_model.joblib"
)

# =====================================
# 3. ĐỊNH NGHĨA DỮ LIỆU ĐẦU VÀO
# =====================================

class ModerationRequest(BaseModel):
    title: str
    description: str

# =====================================
# 4. API KIỂM TRA HOẠT ĐỘNG
# =====================================

@app.get("/")
def home():
    return {
        "message": "KiemJob AI Moderation API is running!",
        "version": "2.0.0",
        "method": "chunk_based_classification"
    }

# =====================================
# 5. API KIỂM DUYỆT
# =====================================

@app.post("/predict")
def predict(request: ModerationRequest):
    return moderate_job_post(
        title=request.title,
        description=request.description,
        vectorizer=vectorizer,
        model=model,
        threshold=0.5
    )
