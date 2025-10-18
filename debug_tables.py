#!/usr/bin/env python3
"""
Veritabanı tablolarını debug etmek için script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal, PredictionRecord, Appointment
from sqlalchemy import text

def debug_tables():
    db = SessionLocal()
    try:
        print("🔍 Veritabanı tablolarını kontrol ediliyor...")
        
        # PredictionRecord sayısı
        pred_count = db.query(PredictionRecord).count()
        print(f"📊 PredictionRecord sayısı: {pred_count}")
        
        # Appointment sayısı
        app_count = db.query(Appointment).count()
        print(f"📋 Appointment sayısı: {app_count}")
        
        if pred_count > 0:
            print("\n📝 İlk PredictionRecord örneği:")
            sample = db.query(PredictionRecord).first()
            print(f"  - ID: {sample.id}")
            print(f"  - Age: {sample.age}")
            print(f"  - Risk Score: {sample.risk_score}")
            print(f"  - Prediction: {sample.prediction}")
            print(f"  - Probability: {sample.probability}")
            print(f"  - Model ID: {sample.model_id}")
            
            # Alanları kontrol et
            print(f"\n🔍 PredictionRecord alanları:")
            for key, value in sample.__dict__.items():
                if not key.startswith('_'):
                    print(f"  - {key}: {type(value).__name__} = {value}")
        else:
            print("❌ PredictionRecord tablosu boş!")
            
        # Appointment tablosunu kontrol et
        if app_count > 0:
            print(f"\n📋 İlk Appointment örneği:")
            app_sample = db.query(Appointment).first()
            print(f"  - ID: {app_sample.id}")
            print(f"  - Prediction Record ID: {app_sample.prediction_record_id}")
            print(f"  - Risk Score: {app_sample.risk_score}")
            print(f"  - Priority Score: {app_sample.priority_score}")
            print(f"  - Status: {app_sample.status}")
            print(f"  - Queue Position: {app_sample.queue_position}")
        
        # Tablo yapısını kontrol et
        print(f"\n🏗️ Tablo yapısı kontrol ediliyor...")
        
        # PredictionRecord tablosu yapısı
        result = db.execute(text("PRAGMA table_info(prediction_records)"))
        columns = result.fetchall()
        print(f"📊 prediction_records kolonları:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
            
        # Appointment tablosu yapısı
        result = db.execute(text("PRAGMA table_info(appointments)"))
        columns = result.fetchall()
        print(f"📋 appointments kolonları:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    debug_tables()
