import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from pymongo import MongoClient

# ========================================
# CẤU HÌNH CƠ SỞ DỮ LIỆU & THƯ MỤC
# ========================================
MONGO_URI = 'mongodb://localhost:27017/'
DB_NAME = 'face_attendance'
COL_NAME = 'students'

# Đảm bảo đường dẫn này trỏ đến thư mục chứa ảnh sinh viên của bạn
# Cấu trúc thư mục phải là: students/<MãSV_TênSV>/1.jpg
STUDENTS_DIR = r'G:\My Drive\MTCNN&FaceNet\students'

def process_and_save():
    # 1. Kết nối DB
    print('⏳ Đang kết nối MongoDB...')
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command('ping')
    except Exception as e:
        print(f"❌ Không thể kết nối MongoDB. Hãy chắc chắn MongoDB đang chạy ở port 27017.\nChi tiết lỗi: {e}")
        return

    db = client[DB_NAME]
    col = db[COL_NAME]
    col.create_index("student_id", unique=True)
    print('✅ Kết nối MongoDB thành công!')

    # 2. Khởi tạo Models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️ Đang chạy AI trên: {device}')
    
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 3. Quét ảnh
    if not os.path.exists(STUDENTS_DIR):
        print(f"❌ Không tìm thấy thư mục {STUDENTS_DIR}. Vui lòng tạo thư mục và thêm ảnh.")
        return

    folders = [f for f in os.listdir(STUDENTS_DIR) if os.path.isdir(os.path.join(STUDENTS_DIR, f))]
    if not folders:
        print("⚠️ Thư mục students trống. Vui lòng thêm các thư mục sinh viên (VD: SV001_NguyenVanA).")
        return

    for folder_name in folders:
        # Tách mã sinh viên. Giả sử format folder: "SV001_Nguyen_Van_A"
        parts = folder_name.split('_', 1)
        student_id = parts[0]
        student_name = parts[1].replace('_', ' ') if len(parts) > 1 else "Unknown"
        
        person_dir = os.path.join(STUDENTS_DIR, folder_name)
        images = [img for img in os.listdir(person_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"⚠️ Bỏ qua {folder_name} vì không có ảnh.")
            continue
            
        embeddings = []
        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    emb = resnet(face.unsqueeze(0)).detach().cpu().numpy().flatten()
                    embeddings.append(emb)
            except Exception as e:
                print(f"❌ Lỗi đọc ảnh {img_path}: {e}")
        
        if embeddings:
            # Lấy trung bình vector nếu sinh viên có nhiều ảnh (tăng độ chính xác)
            avg_emb = np.mean(embeddings, axis=0)
            
            # Lưu vào MongoDB
            student_data = {
                "student_id": student_id,
                "name": student_name,
                "embedding": avg_emb.tolist()
            }
            
            col.update_one(
                {"student_id": student_id},
                {"$set": student_data},
                upsert=True
            )
            print(f"✅ Đã đăng ký/Cập nhật: {student_name} ({student_id})")
        else:
            print(f"❌ Không tìm thấy khuôn mặt hợp lệ nào cho: {folder_name}")

if __name__ == "__main__":
    process_and_save()
    print("🎉 Hoàn tất quá trình trích xuất và đăng ký đặc trưng khuôn mặt!")
