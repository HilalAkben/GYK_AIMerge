#!/usr/bin/env python3
"""
Appointments tablosuna doctor_id sütunu ekleme scripti
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from api.core.config import settings

def add_doctor_id_column():
    """Appointments tablosuna doctor_id sütunu ekler"""
    
    # Veritabanı bağlantısı
    engine = create_engine(settings.database_url)
    
    try:
        with engine.connect() as conn:
            # Önce sütunun var olup olmadığını kontrol et
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'appointments' 
                AND column_name = 'doctor_id'
            """))
            
            if result.fetchone():
                print("doctor_id sütunu zaten mevcut!")
                return
            
            # doctor_id sütununu ekle
            conn.execute(text("""
                ALTER TABLE appointments 
                ADD COLUMN doctor_id INTEGER REFERENCES users(id)
            """))
            
            conn.commit()
            print("✅ doctor_id sütunu başarıyla eklendi!")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Appointments tablosuna doctor_id sütunu ekleniyor...")
    success = add_doctor_id_column()
    if success:
        print("✅ İşlem tamamlandı!")
    else:
        print("❌ İşlem başarısız!")
