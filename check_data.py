"""
Veri Kontrol Scripti
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from api.core.database import SessionLocal, PredictionRecord, Appointment

def check_data():
    """Veritabanındaki verileri kontrol et."""
    db = SessionLocal()
    try:
        # PredictionRecord sayısını kontrol et
        prediction_count = db.query(PredictionRecord).count()
        print(f"📊 Toplam PredictionRecord sayısı: {prediction_count}")
        
        if prediction_count > 0:
            # İlk 5 kaydı göster
            records = db.query(PredictionRecord).order_by(
                PredictionRecord.risk_score.desc()
            ).limit(5).all()
            
            print("\n🔍 En yüksek risk skorlu 5 kayıt:")
            for i, record in enumerate(records, 1):
                print(f"  {i}. ID: {record.id}, Risk: {record.risk_score:.3f}, Model: {record.best_model_name}")
        
        # Appointment sayısını kontrol et
        appointment_count = db.query(Appointment).count()
        print(f"\n📋 Toplam Appointment sayısı: {appointment_count}")
        
        if appointment_count > 0:
            # İlk 5 randevuyu göster
            appointments = db.query(Appointment).order_by(
                Appointment.queue_position.asc()
            ).limit(5).all()
            
            print("\n🔍 İlk 5 randevu:")
            for appointment in appointments:
                print(f"  Sıra: {appointment.queue_position}, ID: {appointment.id}, Risk: {appointment.risk_score:.3f}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_data()
