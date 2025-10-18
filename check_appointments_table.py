#!/usr/bin/env python3
"""
Appointments tablosunun yapısını kontrol etme scripti
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from api.core.config import settings

def check_appointments_table():
    """Appointments tablosunun yapısını kontrol eder"""
    
    # Veritabanı bağlantısı
    engine = create_engine(settings.database_url)
    
    try:
        with engine.connect() as conn:
            # Tablo yapısını kontrol et
            result = conn.execute(text("""
                SELECT column_name, data_type, is_nullable 
                FROM information_schema.columns 
                WHERE table_name = 'appointments' 
                ORDER BY ordinal_position
            """))
            
            print("Appointments tablosu sütunları:")
            print("-" * 50)
            for row in result:
                print(f"{row[0]:<25} {row[1]:<15} {row[2]}")
            
            # doctor_id sütununun var olup olmadığını kontrol et
            result2 = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'appointments' 
                AND column_name = 'doctor_id'
            """))
            
            if result2.fetchone():
                print("\n✅ doctor_id sütunu mevcut!")
            else:
                print("\n❌ doctor_id sütunu mevcut değil!")
                
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    check_appointments_table()
