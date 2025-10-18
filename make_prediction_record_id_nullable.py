import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal
from sqlalchemy import text

def make_prediction_record_id_nullable():
    """
    appointments tablosundaki prediction_record_id sütununu nullable yap
    """
    db = SessionLocal()
    try:
        print("prediction_record_id sütunu nullable yapiliyor...")
        
        # NOT NULL constraint'i kaldır
        db.execute(text("""
            ALTER TABLE appointments 
            ALTER COLUMN prediction_record_id DROP NOT NULL
        """))
        
        db.commit()
        print("SUCCESS: prediction_record_id sütunu artık nullable!")
        
    except Exception as e:
        print(f"ERROR: Hata olustu: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    make_prediction_record_id_nullable()
