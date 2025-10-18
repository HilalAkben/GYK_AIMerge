#!/usr/bin/env python3
"""
PostgreSQL için appointment tablosunu düzelt
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal
from sqlalchemy import text

def fix_appointments_table():
    """Appointment tablosunu düzelt"""
    db = SessionLocal()
    try:
        print("🔧 Appointment tablosu düzeltiliyor...")
        
        # Eksik kolonları ekle (PostgreSQL syntax)
        alter_queries = [
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS age INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS gender INTEGER", 
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS height DECIMAL(6,2)",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS weight DECIMAL(6,2)",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS ap_hi INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS ap_lo INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS cholesterol INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS gluc INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS smoke INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS alco INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS active INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS risk_score DECIMAL(10,4)",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS priority_score DECIMAL(10,4)",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS prediction_label INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS prediction_probability DECIMAL(10,4)",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS model_name VARCHAR(100)",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'pending'",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS priority VARCHAR(20) DEFAULT 'normal'",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS queue_position INTEGER",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS appointment_date TIMESTAMP",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS assigned_at TIMESTAMP",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            "ALTER TABLE appointments ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ]
        
        for query in alter_queries:
            try:
                db.execute(text(query))
                print(f"✅ {query}")
            except Exception as e:
                print(f"⚠️ {query} - {e}")
        
        db.commit()
        print("✅ Appointment tablosu düzeltildi!")
        
        # Tablo yapısını kontrol et
        result = db.execute(text("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = 'appointments' 
            ORDER BY ordinal_position
        """))
        columns = result.fetchall()
        
        print("\n📋 Appointment tablosu yapısı:")
        for col in columns:
            print(f"  - {col[0]} ({col[1]}) - Nullable: {col[2]} - Default: {col[3]}")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    fix_appointments_table()
