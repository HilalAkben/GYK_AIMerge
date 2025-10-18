"""
Manuel Randevu Atama Scripti
İlk 20 hastayı manuel olarak randevu tablosuna atar
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from api.core.database import SessionLocal, PredictionRecord, Appointment
from sqlalchemy import desc
from datetime import datetime

def assign_appointments():
    """Manuel olarak randevu atama işlemini gerçekleştir."""
    db = SessionLocal()
    try:
        print("🔄 Randevu atama işlemi başlatılıyor...")
        
        # Mevcut randevu kayıtlarını temizle
        deleted_count = db.query(Appointment).count()
        db.query(Appointment).delete()
        db.commit()
        print(f"🗑️ {deleted_count} eski randevu kaydı silindi")
        
        # En yüksek risk skoruna sahip 20 hastayı getir
        top_patients = db.query(PredictionRecord).order_by(
            desc(PredictionRecord.risk_score),
            desc(PredictionRecord.priority_score),
            desc(PredictionRecord.created_at)
        ).limit(20).all()
        
        print(f"📊 {len(top_patients)} hasta bulundu")
        
        if len(top_patients) == 0:
            print("❌ Hiç hasta bulunamadı!")
            return
        
        # Her hasta için randevu kaydı oluştur
        for i, patient in enumerate(top_patients, 1):
            appointment = Appointment(
                prediction_record_id=patient.id,
                age=patient.age,
                gender=patient.gender,
                height=patient.height,
                weight=patient.weight,
                ap_hi=patient.ap_hi,
                ap_lo=patient.ap_lo,
                cholesterol=patient.cholesterol,
                gluc=patient.gluc,
                smoke=patient.smoke,
                alco=patient.alco,
                active=patient.active,
                risk_score=patient.risk_score,
                priority_score=patient.priority_score,
                prediction_label=patient.prediction_label,
                prediction_probability=patient.prediction_probability,
                model_name=patient.best_model_name,
                queue_position=i,
                status="pending",
                priority=_determine_priority(patient.risk_score),
                assigned_at=datetime.utcnow()
            )
            db.add(appointment)
            print(f"✅ Hasta #{patient.id} - Risk: {patient.risk_score:.3f} - Sıra: {i}")
        
        db.commit()
        print(f"🎉 {len(top_patients)} hasta başarıyla randevu tablosuna atandı!")
        
        # Kontrol et
        appointment_count = db.query(Appointment).count()
        print(f"📋 Toplam randevu sayısı: {appointment_count}")
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        db.rollback()
    finally:
        db.close()

def _determine_priority(risk_score: float) -> str:
    """Risk skoruna göre öncelik belirle"""
    if risk_score >= 0.8:
        return "urgent"
    elif risk_score >= 0.6:
        return "high"
    elif risk_score >= 0.4:
        return "normal"
    else:
        return "low"

if __name__ == "__main__":
    assign_appointments()
