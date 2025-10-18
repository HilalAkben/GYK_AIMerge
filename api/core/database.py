"""
Veritabanı Modülü
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, DECIMAL, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import json
from api.core.config import settings

# Veritabanı engine oluştur
engine = create_engine(
    settings.database_url,
    echo=settings.debug  # SQL sorgularını görmek için
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class
Base = declarative_base()

class User(Base):
    """Kullanıcı tablosu"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(20), default="user")  # user, admin, doctor
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    patients = relationship("Patient", back_populates="user")
    risk_assessments = relationship("RiskAssessment", back_populates="user")
    doctor_targets = relationship("DoctorTarget", foreign_keys="DoctorTarget.doctor_id", back_populates="doctor")

class Patient(Base):
    """Hasta bilgileri tablosu"""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_code = Column(String(20), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    height = Column(DECIMAL(5,2), nullable=False)
    weight = Column(DECIMAL(5,2), nullable=False)
    systolic_bp = Column(Integer, nullable=False)
    diastolic_bp = Column(Integer, nullable=False)
    cholesterol = Column(Integer, nullable=False)
    glucose = Column(Integer, nullable=False)
    smoker = Column(Boolean, default=False)
    alcoholic = Column(Boolean, default=False)
    physically_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="patients")
    risk_assessments = relationship("RiskAssessment", back_populates="patient")

class RiskAssessment(Base):
    """Risk değerlendirme tablosu"""
    __tablename__ = "risk_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    model_used = Column(String(50), nullable=False)
    risk_score = Column(DECIMAL(5,2), nullable=False)
    risk_level = Column(String(20), nullable=False)
    confidence_score = Column(DECIMAL(4,3))
    prediction_probability = Column(DECIMAL(4,3))
    raw_data = Column(JSON, nullable=False)
    notes = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="risk_assessments")
    user = relationship("User", back_populates="risk_assessments")

class ModelRecord(Base):
    """Eğitilmiş modellerin kayıtları."""
    __tablename__ = "model_records"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # 'base', 'ensemble', 'tuned'
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=True)
    model_path = Column(String(255), nullable=False)
    feature_names = Column(Text, nullable=True)  # JSON string
    training_params = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def to_dict(self):
        """Model kaydını dictionary'e çevir."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "model_path": self.model_path,
            "feature_names": json.loads(self.feature_names) if self.feature_names else None,
            "training_params": json.loads(self.training_params) if self.training_params else None,
            "created_at": self.created_at.isoformat(),
            "is_active": self.is_active
        }

class PredictionRecord(Base):
    """Tahmin kayıtları."""
    __tablename__ = "prediction_records"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False, index=True)

    # Snapshot alanları (JSON içindeki verileri kolon bazında da tut)
    patient_external_id = Column(Integer)  # PostgreSQL BIGINT ile uyumlu (SQLAlchemy BigInteger'a gerek duyulursa güncellenebilir)
    age = Column(Integer)
    gender = Column(Integer)
    height = Column(DECIMAL(6, 2))
    weight = Column(DECIMAL(6, 2))
    ap_hi = Column(Integer)
    ap_lo = Column(Integer)
    cholesterol = Column(Integer)
    gluc = Column(Integer)
    smoke = Column(Boolean)
    alco = Column(Boolean)
    active = Column(Boolean)

    # Geriye dönük uyumluluk için tam giriş JSON'u
    patient_data = Column(Text, nullable=False)  # JSON string
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class PatientRiskRecord(Base):
    """Kullanıcı girdileri ve tahmin skorlarının izlendiği tablo."""
    __tablename__ = "patient_risk_records"

    id = Column(Integer, primary_key=True, index=True)
    # Girdiler
    age = Column(Integer, nullable=False)
    gender = Column(Integer, nullable=False)
    height = Column(Float, nullable=False)
    weight = Column(Float, nullable=False)
    ap_hi = Column(Integer, nullable=False)
    ap_lo = Column(Integer, nullable=False)
    cholesterol = Column(Integer, nullable=False)
    gluc = Column(Integer, nullable=False)
    smoke = Column(Boolean, nullable=False)
    alco = Column(Boolean, nullable=False)
    active = Column(Boolean, nullable=False)

    # Model ve skorlar
    best_model_name = Column(String(100), nullable=False)
    risk_score = Column(Float, nullable=False)
    weighted_factors_score = Column(Float, nullable=False)
    priority_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DoctorTarget(Base):
    """Doktorun hasta target değerlerini belirlediği tablo"""
    __tablename__ = "doctor_targets"
    
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    prediction_record_id = Column(Integer, ForeignKey("prediction_records.id"), nullable=False)
    
    # Hasta input değerleri (PredictionRecord'dan kopyalanır)
    age = Column(Integer, nullable=False)
    gender = Column(Integer, nullable=False)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    ap_hi = Column(Integer, nullable=False)
    ap_lo = Column(Integer, nullable=False)
    cholesterol = Column(Integer, nullable=False)
    gluc = Column(Integer, nullable=False)
    smoke = Column(Integer, nullable=False)
    alco = Column(Integer, nullable=False)
    active = Column(Integer, nullable=False)
    
    # Doktorun belirlediği target değer
    target_value = Column(Integer, nullable=False)  # 0: Kalp hastalığı yok, 1: Kalp hastalığı var
    
    # Doktorun notları
    doctor_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # İlişkiler
    doctor = relationship("User", foreign_keys=[doctor_id], back_populates="doctor_targets")
    patient = relationship("User", foreign_keys=[patient_id])
    prediction_record = relationship("PredictionRecord")

class Appointment(Base):
    """Randevu tablosu - En yüksek riskli 20 hasta"""
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Hasta ID'si
    prediction_record_id = Column(Integer, ForeignKey("prediction_records.id"), nullable=True, unique=True)
    
    # Hasta bilgileri (PredictionRecord'dan kopyalanır)
    age = Column(Integer, nullable=False)
    gender = Column(Integer, nullable=False)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    ap_hi = Column(Integer, nullable=False)
    ap_lo = Column(Integer, nullable=False)
    cholesterol = Column(Integer, nullable=False)
    gluc = Column(Integer, nullable=False)
    smoke = Column(Integer, nullable=False)
    alco = Column(Integer, nullable=False)
    active = Column(Integer, nullable=False)
    
    # Risk ve öncelik bilgileri
    risk_score = Column(Float, nullable=False)
    priority_score = Column(Float, nullable=False)
    prediction_label = Column(Integer, nullable=False)
    prediction_probability = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    
    # Doktor atama
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Atanan doktor ID'si
    
    # Randevu durumu
    status = Column(String(20), default="pending")  # pending, assigned, completed, cancelled
    priority = Column(String(20), default="normal")  # low, normal, high, urgent
    
    # Randevu sırası
    queue_position = Column(Integer, nullable=True)  # Sıradaki pozisyon
    
    # Randevu tarihi ve saati
    appointment_date = Column(DateTime, nullable=True)
    assigned_at = Column(DateTime, nullable=True)  # Randevuya atandığı tarih
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # İlişkiler
    patient = relationship("User", foreign_keys=[patient_id])
    doctor = relationship("User", foreign_keys=[doctor_id])
    prediction_record = relationship("PredictionRecord")
    doctor_diagnoses = relationship("DoctorDiagnosis", back_populates="appointment")

class DoctorDiagnosis(Base):
    """Doktor teşhis tablosu - Target değerleri ile birlikte"""
    __tablename__ = "doctor_diagnoses"
    
    id = Column(Integer, primary_key=True, index=True)
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Hasta bilgileri (Appointment'ten kopyalanır)
    age = Column(Integer, nullable=False)
    gender = Column(Integer, nullable=False)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    ap_hi = Column(Integer, nullable=False)
    ap_lo = Column(Integer, nullable=False)
    cholesterol = Column(Integer, nullable=False)
    gluc = Column(Integer, nullable=False)
    smoke = Column(Integer, nullable=False)
    alco = Column(Integer, nullable=False)
    active = Column(Integer, nullable=False)
    
    # Risk bilgileri
    risk_score = Column(Float, nullable=False)
    priority_score = Column(Float, nullable=False)
    prediction_label = Column(Integer, nullable=False)
    prediction_probability = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    
    # Doktorun belirlediği target değer
    doctor_target = Column(Integer, nullable=False)  # 0: Kalp hastalığı yok, 1: Kalp hastalığı var
    
    # Doktorun notları
    doctor_notes = Column(Text, nullable=True)
    
    # Teşhis durumu
    diagnosis_status = Column(String(20), default="completed")  # completed, pending, reviewed
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # İlişkiler
    appointment = relationship("Appointment", back_populates="doctor_diagnoses")
    doctor = relationship("User", foreign_keys=[doctor_id])

async def init_database():
    """Veritabanını başlat."""
    # Tabloları oluştur
    Base.metadata.create_all(bind=engine)
    print("Veritabani baslatildi")

def get_db():
    """Veritabanı session'ı al."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
