#!/usr/bin/env python3
"""
PostgreSQL için randevu atama sistemi
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal, PredictionRecord, Appointment
from sqlalchemy import text
from datetime import datetime

def setup_postgres_appointments():
    """PostgreSQL için randevu atama sistemi kur"""
    db = SessionLocal()
    try:
        print("🔧 PostgreSQL randevu sistemi kuruluyor...")
        
        # Önce tabloyu düzelt (PostgreSQL syntax)
        print("📋 Appointment tablosu düzeltiliyor...")
        
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
        
        # Manuel randevu atama işlemi
        print("\n🔄 Manuel randevu atama işlemi...")
        
        # Mevcut randevuları temizle
        db.execute(text("DELETE FROM appointments"))
        db.commit()
        
        # En yüksek riskli 20 hastayı seç ve randevu tablosuna ekle
        insert_query = """
        INSERT INTO appointments (
            prediction_record_id, age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active, risk_score, priority_score,
            prediction_label, prediction_probability, model_name, status, priority,
            queue_position, appointment_date, assigned_at, created_at, updated_at
        )
        SELECT
            pr.id AS prediction_record_id,
            pr.age, pr.gender, pr.height, pr.weight, pr.ap_hi, pr.ap_lo,
            pr.cholesterol, pr.gluc, pr.smoke, pr.alco, pr.active,
            pr.risk_score, pr.risk_score AS priority_score,
            pr.prediction AS prediction_label,
            pr.probability AS prediction_probability,
            CONCAT('Model_', pr.model_id) AS model_name,
            'pending' AS status,
            CASE
                WHEN pr.risk_score >= 0.8 THEN 'urgent'
                WHEN pr.risk_score >= 0.6 THEN 'high'
                WHEN pr.risk_score >= 0.4 THEN 'normal'
                ELSE 'low'
            END AS priority,
            ROW_NUMBER() OVER (ORDER BY pr.risk_score DESC) AS queue_position,
            NOW() + INTERVAL '1 day' AS appointment_date,
            NOW() AS assigned_at,
            NOW() AS created_at,
            NOW() AS updated_at
        FROM prediction_records pr
        ORDER BY pr.risk_score DESC
        LIMIT 20
        """
        
        db.execute(text(insert_query))
        db.commit()
        
        # Sonuçları kontrol et
        result = db.execute(text("SELECT COUNT(*) FROM appointments"))
        count = result.scalar()
        print(f"✅ {count} hasta randevu tablosuna atandı!")
        
        # Randevu detaylarını göster
        result = db.execute(text("""
            SELECT 
                id, age, gender, risk_score, priority, queue_position, 
                prediction_label, model_name, status
            FROM appointments 
            ORDER BY queue_position 
            LIMIT 5
        """))
        appointments = result.fetchall()
        
        print("\n📋 İlk 5 randevu:")
        for app in appointments:
            print(f"  - ID: {app[0]}, Yaş: {app[1]}, Cinsiyet: {app[2]}, Risk: {app[3]:.3f}, Öncelik: {app[4]}, Sıra: {app[5]}, Tahmin: {app[6]}, Model: {app[7]}, Durum: {app[8]}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    setup_postgres_appointments()
