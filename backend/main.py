import io
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1

app = FastAPI(title="Hệ thống Nhận diện Khuôn mặt (Backend)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# 1. CẤU HÌNH NGƯỠNG TỪ NGHIÊN CỨU (JSON)
# ========================================
THRESHOLD = 1.0  # Ranh giới phân biệt cùng người và khác người

# ========================================
# 2. DATABASE & AI MODELS
# ========================================
MONGO_URI = 'mongodb://localhost:27017/'
client = MongoClient(MONGO_URI)
db = client['face_attendance']
col = db['students']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_euclidean_distance(emb1, emb2):
    return np.linalg.norm(np.array(emb1) - np.array(emb2))

# ========================================
# 3. API ENDPOINTS
# ========================================
@app.get("/")
def read_root():
    return {"message": "API Nhận Diện Khuôn Mặt Đang Chạy! (Threshold: 1.0)"}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    """
    Nhận 1 bức ảnh từ hệ thống (camera), bóc tách khuôn mặt và so sánh với MongoDB.
    """
    try:
        # Đọc ảnh
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Cắt khuôn mặt
        face = mtcnn(img)
        if face is None:
            return {"status": "error", "message": "Không tìm thấy khuôn mặt trong ảnh!"}
            
        # Tạo vector 512 chiều
        emb = resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten()
        
        # So sánh với Database
        students = list(col.find())
        if not students:
            return {"status": "error", "message": "Database trống. Vui lòng chạy batch_register.py trước!"}
            
        best_match = None
        min_dist = float('inf')
        
        # Tìm người giống nhất
        for student in students:
            db_emb = student.get("embedding")
            if not db_emb:
                continue
                
            dist = get_euclidean_distance(emb, db_emb)
            if dist < min_dist:
                min_dist = dist
                best_match = student
                
        # Ra quyết định bằng THRESHOLD
        if min_dist <= THRESHOLD and best_match:
            return {
                "status": "success",
                "message": "Điểm danh thành công!",
                "student_id": best_match["student_id"],
                "name": best_match["name"],
                "distance": round(float(min_dist), 4),
                "threshold": THRESHOLD
            }
        else:
            return {
                "status": "fail",
                "message": "Khuôn mặt lạ hoặc không chắc chắn (Khoảng cách quá xa).",
                "distance": round(float(min_dist), 4),
                "threshold": THRESHOLD
            }
            
    except Exception as e:
        return {"status": "error", "message": f"Lỗi hệ thống: {str(e)}"}
