import os
import pandas as pd
import sqlite3

# 1. Đường dẫn chính xác tới file CSV (Bỏ "data/")
csv_path = "healthcare-dataset-stroke-data.csv"

if not os.path.exists(csv_path):
    print(f"❌ Không tìm thấy file CSV tại: {csv_path}")
else:
    print("⏳ Đang xử lý chuyển đổi...")
    # 2. Đọc file CSV bằng Pandas
    df = pd.read_csv(csv_path)
    
    # 3. Tạo kết nối thẳng vào file stroke.db (Đổi tên thành stroke.db cho đồng bộ với file Notebook)
    db_path = "stroke.db" 
    conn = sqlite3.connect(db_path)
    
    # 4. Đẩy toàn bộ bảng vào database
    df.to_sql("stroke_table", conn, index=False, if_exists="replace")
    
    # Đóng kết nối
    conn.close()
    print(f"🎉 ĐÃ TẠO FILE DATABASE THÀNH CÔNG TẠI: {db_path}")