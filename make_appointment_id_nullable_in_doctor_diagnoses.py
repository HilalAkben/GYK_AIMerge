import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal
from sqlalchemy import text

def make_appointment_id_nullable():
    """
    doctor_diagnoses tablosundaki appointment_id sütununu nullable yap.
    """
    db = SessionLocal()
    try:
        print("doctor_diagnoses tablosundaki appointment_id sütunu nullable yapiliyor...")
        
        # appointment_id sütununu nullable yap
        db.execute(text("""
            ALTER TABLE doctor_diagnoses 
            ALTER COLUMN appointment_id DROP NOT NULL
        """))
        
        db.commit()
        print("SUCCESS: appointment_id sütunu artık nullable!")
        
    except Exception as e:
        print(f"ERROR: Hata oluştu: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    make_appointment_id_nullable()
