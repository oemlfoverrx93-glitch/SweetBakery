import os
from sqlalchemy import create_engine

def get_engine():
    # Lấy đường dẫn tuyệt đối đến thư mục 'code'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Định vị file stroke_data.db nằm ngay trong thư mục 'code'
    db_path = os.path.join(current_dir, "stroke.db")
    
    # Trả về engine kết nối SQLite
    return create_engine(f"sqlite:///{db_path}")