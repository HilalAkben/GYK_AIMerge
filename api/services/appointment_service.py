"""
Randevu Atama Servisi
Her 10 dakikada bir en yüksek riskli 20 hastayı randevu tablosuna atar
"""

import asyncio
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc
from api.core.database import SessionLocal, PredictionRecord, Appointment, User
from api.core.config import settings
import logging

logger = logging.getLogger(__name__)

class AppointmentService:
    def __init__(self):
        self.is_running = False
        self.task = None
    
    async def start(self):
        """Servisi başlat"""
        if not self.is_running:
            self.is_running = True
            self.task = asyncio.create_task(self._run_periodic_assignment())
            logger.info("Randevu atama servisi baslatildi")
    
    async def stop(self):
        """Servisi durdur"""
        if self.is_running:
            self.is_running = False
            if self.task:
                self.task.cancel()
                try:
                    await self.task
                except asyncio.CancelledError:
                    pass
            logger.info("Randevu atama servisi durduruldu")
    
    async def _run_periodic_assignment(self):
        """Her 10 dakikada bir randevu atama işlemini çalıştır"""
        while self.is_running:
            try:
                await self.assign_top_risk_patients()
                logger.info("Randevu atama islemi tamamlandi")
            except Exception as e:
                logger.error(f"Randevu atama hatasi: {e}")
            
            # 10 dakika bekle
            await asyncio.sleep(600)  # 600 saniye = 10 dakika
    
    async def assign_top_risk_patients(self):
        """En yüksek riskli 20 hastayı randevu tablosuna ata"""
        db = SessionLocal()
        try:
            # Mevcut randevu kayıtlarını temizle (Foreign Key hatasını önlemek için DoctorDiagnosis'ları da sil)
            from api.core.database import DoctorDiagnosis
            db.query(DoctorDiagnosis).delete()
            db.query(Appointment).delete()
            db.commit()
            
            # En yüksek risk skoruna sahip 20 hastayı getir
            top_patients = db.query(PredictionRecord).order_by(
                desc(PredictionRecord.risk_score),
                desc(PredictionRecord.created_at)
            ).limit(20).all()
            
            logger.info(f"{len(top_patients)} hasta randevu tablosuna ataniyor...")
            
            # Her hasta için randevu kaydı oluştur
            for i, patient in enumerate(top_patients, 1):
                # patient_external_id'yi hash'leyerek küçük ID oluştur
                patient_id = abs(hash(str(patient.patient_external_id))) % 1000000  # 1 milyon altında ID
                
                # Kullanıcı oluştur veya bul
                user = db.query(User).filter(User.id == patient_id).first()
                if not user:
                    # Kullanıcı yoksa oluştur
                    user = User(
                        id=patient_id,
                        username=f"patient_{patient_id}",
                        email=f"patient_{patient_id}@example.com",
                        password_hash="dummy_hash_for_patient",  # Dummy password hash
                        full_name=f"Hasta {patient_id}",
                        role="patient",
                        is_active=True
                    )
                    db.add(user)
                    db.flush()  # ID'yi almak için flush
                
                appointment = Appointment(
                    patient_id=user.id,  # Gerçek kullanıcı ID'sini kullan
                    prediction_record_id=patient.id,
                    age=patient.age,
                    gender=patient.gender,
                    height=patient.height,
                    weight=patient.weight,
                    ap_hi=patient.ap_hi,
                    ap_lo=patient.ap_lo,
                    cholesterol=patient.cholesterol,
                    gluc=patient.gluc,
                    smoke=int(patient.smoke),
                    alco=int(patient.alco),
                    active=int(patient.active),
                    risk_score=patient.risk_score,
                    priority_score=patient.risk_score,  # Risk skorunu priority olarak kullan
                    prediction_label=patient.prediction,
                    prediction_probability=patient.probability,
                    model_name=f"Model_{patient.model_id}",
                    queue_position=i,  # Sıra pozisyonu
                    status="pending",
                    priority=self._determine_priority(patient.risk_score),
                    assigned_at=datetime.utcnow()
                )
                db.add(appointment)
            
            db.commit()
            logger.info(f"{len(top_patients)} hasta basariyla randevu tablosuna atandi")
            
        except Exception as e:
            logger.error(f"Randevu atama hatasi: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    def _determine_priority(self, risk_score: float) -> str:
        """Risk skoruna göre öncelik belirle"""
        if risk_score >= 0.8:
            return "urgent"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "normal"
        else:
            return "low"
    
    async def get_current_appointments(self) -> list:
        """Mevcut randevu listesini getir"""
        db = SessionLocal()
        try:
            appointments = db.query(Appointment).order_by(
                Appointment.queue_position.asc()
            ).all()
            return appointments
        finally:
            db.close()
    
    async def get_appointment_by_id(self, appointment_id: int) -> Appointment:
        """ID'ye göre randevu getir"""
        db = SessionLocal()
        try:
            return db.query(Appointment).filter(Appointment.id == appointment_id).first()
        finally:
            db.close()

# Global servis instance
appointment_service = AppointmentService()
