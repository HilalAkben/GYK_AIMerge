"""
Doktor Panel API Routes
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session, joinedload
from typing import List, Dict, Any
from api.core.database import get_db, User, PredictionRecord, DoctorTarget, Appointment, DoctorDiagnosis, ModelRecord
from api.services.auth_service import require_doctor
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import os
from pathlib import Path

router = APIRouter()

def get_model_display_name(model_name: str, db: Session) -> str:
    """
    Model ismini model_records tablosundan çek.
    """
    if not model_name:
        return "Bilinmeyen Model"
    
    try:
        # Model_44 formatındaysa ID'yi çıkar
        if model_name.startswith("Model_"):
            model_id = model_name.replace("Model_", "")
            try:
                model_id_int = int(model_id)
                # model_records tablosundan gerçek model ismini çek
                model_record = db.query(ModelRecord).filter(ModelRecord.id == model_id_int).first()
                if model_record:
                    return model_record.model_name
            except ValueError:
                pass
        
        # Direkt model_name ile ara
        model_record = db.query(ModelRecord).filter(ModelRecord.model_name == model_name).first()
        if model_record:
            return model_record.model_name
        
        # Bulunamazsa orijinal ismi döndür
        return model_name
        
    except Exception as e:
        # Hata durumunda orijinal ismi döndür
        return model_name

def append_to_csv(diagnosis_data: Dict[str, Any]):
    """
    Doktor teşhisini cardio_train.csv dosyasına ekle.
    """
    try:
        # CSV dosyasının yolu
        csv_path = Path(__file__).parent.parent.parent / "data" / "raw" / "cardio_train.csv"
        
        # Mevcut CSV dosyasını oku
        if csv_path.exists():
            df = pd.read_csv(csv_path, sep=';')
        else:
            # CSV dosyası yoksa yeni oluştur
            df = pd.DataFrame(columns=['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
                                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio'])
        
        # Yeni kayıt için veri hazırla
        # Yaşı güne çevir (yıl * 365)
        age_in_days = diagnosis_data['age'] * 365
        
        new_record = {
            'id': len(df),  # Yeni ID (mevcut satır sayısı)
            'age': age_in_days,  # Yaş gün formatında
            'gender': diagnosis_data['gender'],
            'height': diagnosis_data['height'],
            'weight': diagnosis_data['weight'],
            'ap_hi': diagnosis_data['ap_hi'],
            'ap_lo': diagnosis_data['ap_lo'],
            'cholesterol': diagnosis_data['cholesterol'],
            'gluc': diagnosis_data['gluc'],
            'smoke': diagnosis_data['smoke'],
            'alco': diagnosis_data['alco'],
            'active': diagnosis_data['active'],
            'cardio': diagnosis_data['doctor_target']  # doctor_target -> cardio olarak kaydet
        }
        
        # Yeni kaydı DataFrame'e ekle
        new_df = pd.DataFrame([new_record])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # CSV dosyasına kaydet
        df.to_csv(csv_path, sep=';', index=False)
        
        print(f"Doktor teshisi CSV dosyasina kaydedildi: {csv_path}")
        return True
        
    except Exception as e:
        print(f"CSV dosyasina kayit hatasi: {e}")
        return False

class DoctorTargetRequest(BaseModel):
    """Doktor target belirleme isteği"""
    appointment_id: int
    target_value: int  # 0: Kalp hastalığı yok, 1: Kalp hastalığı var
    doctor_notes: str = None

class AppointmentRecord(BaseModel):
    """Randevu kaydı modeli"""
    id: int
    prediction_record_id: int
    age: int
    gender: int
    height: float
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    risk_score: float
    priority_score: float
    prediction_label: int
    prediction_probability: float
    model_name: str
    status: str
    priority: str
    queue_position: int
    assigned_at: datetime
    created_at: datetime
    doctor_diagnosis: Dict[str, Any] = None

class PatientRecord(BaseModel):
    """Hasta kaydı modeli"""
    id: int
    patient_name: str
    patient_email: str
    age: int
    gender: int
    height: float
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    risk_score: float
    priority_score: float
    prediction_label: int
    prediction_probability: float
    model_name: str
    created_at: datetime
    doctor_target: Dict[str, Any] = None

@router.get("/debug", response_model=Dict[str, Any])
async def debug_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Debug bilgileri - veritabanı durumunu kontrol et.
    """
    try:
        # PredictionRecord sayısı
        prediction_count = db.query(PredictionRecord).count()
        
        # Appointment sayısı
        appointment_count = db.query(Appointment).count()
        
        # İlk 5 PredictionRecord
        top_predictions = db.query(PredictionRecord).order_by(
            PredictionRecord.risk_score.desc()
        ).limit(5).all()
        
        prediction_list = []
        for pred in top_predictions:
            prediction_list.append({
                "id": pred.id,
                "risk_score": pred.risk_score,
                "model_name": pred.best_model_name,
                "created_at": pred.created_at.isoformat()
            })
        
        return {
            "success": True,
            "message": "Debug bilgileri",
            "data": {
                "prediction_count": prediction_count,
                "appointment_count": appointment_count,
                "top_predictions": prediction_list,
                "current_user": {
                    "id": current_user.id,
                    "username": current_user.username,
                    "role": current_user.role
                }
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Debug hatası: {str(e)}",
            "data": {}
        }

@router.get("/appointments", response_model=Dict[str, Any])
async def get_appointments(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Doktor için randevu tablosundaki hastaları getir.
    """
    try:
        # Randevu tablosundaki hastaları getir (hasta adı ile birlikte)
        appointments = db.query(Appointment).options(
            joinedload(Appointment.patient)
        ).order_by(
            Appointment.queue_position.asc()
        ).all()
        
        print(f"DEBUG: Bulunan randevu sayısı: {len(appointments)}")
        
        appointment_list = []
        for appointment in appointments:
            # Hasta adını al
            patient_name = "Bilinmeyen Hasta"
            print(f"DEBUG: Appointment {appointment.id} - patient_id: {appointment.patient_id}, patient: {appointment.patient}")
            
            if appointment.patient:
                patient_name = appointment.patient.username  # full_name yerine username kullan
                print(f"DEBUG: Relationship ile hasta adı: {patient_name}")
            elif appointment.patient_id:
                # Eğer relationship çalışmazsa manuel olarak hasta adını al
                patient = db.query(User).filter(User.id == appointment.patient_id).first()
                if patient:
                    patient_name = patient.username  # full_name yerine username kullan
                    print(f"DEBUG: Manuel sorgu ile hasta adı: {patient_name}")
                else:
                    print(f"DEBUG: Hasta bulunamadı - ID: {appointment.patient_id}")
            else:
                print(f"DEBUG: patient_id null - Appointment ID: {appointment.id}")
            
            appointment_data = {
                "id": appointment.id,
                "prediction_record_id": appointment.prediction_record_id,
                "patient_id": appointment.patient_id,
                "patient_name": patient_name,  # Hasta adı eklendi
                "age": appointment.age,
                "gender": appointment.gender,
                "height": float(appointment.height) if appointment.height else None,
                "weight": float(appointment.weight) if appointment.weight else None,
                "ap_hi": appointment.ap_hi,
                "ap_lo": appointment.ap_lo,
                "cholesterol": appointment.cholesterol,
                "gluc": appointment.gluc,
                "smoke": appointment.smoke,
                "alco": appointment.alco,
                "active": appointment.active,
                "risk_score": appointment.risk_score,
                "priority_score": appointment.priority_score,
                "prediction_label": appointment.prediction_label,
                "prediction_probability": appointment.prediction_probability,
                "model_name": get_model_display_name(appointment.model_name, db),
                "status": appointment.status,
                "priority": appointment.priority,
                "queue_position": appointment.queue_position,
                "assigned_at": appointment.assigned_at.isoformat() if appointment.assigned_at else None,
                "created_at": appointment.created_at.isoformat()
            }
            
            # Doktorun daha önce teşhis koyduğu kayıt var mı kontrol et
            doctor_diagnosis = db.query(DoctorDiagnosis).filter(
                DoctorDiagnosis.doctor_id == current_user.id,
                DoctorDiagnosis.appointment_id == appointment.id
            ).first()
            
            if doctor_diagnosis:
                appointment_data["doctor_diagnosis"] = {
                    "id": doctor_diagnosis.id,
                    "doctor_target": doctor_diagnosis.doctor_target,
                    "doctor_notes": doctor_diagnosis.doctor_notes,
                    "diagnosis_status": doctor_diagnosis.diagnosis_status,
                    "created_at": doctor_diagnosis.created_at.isoformat()
                }
            
            appointment_list.append(appointment_data)
        
        return {
            "success": True,
            "message": f"{len(appointment_list)} randevu bulundu",
            "appointments": appointment_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Randevu listesi getirilirken hata oluştu: {str(e)}"
        )

@router.post("/set-target", response_model=Dict[str, Any])
async def set_patient_target(
    target_request: DoctorTargetRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Doktorun randevu için target değerini belirlemesi ve teşhis kaydı oluşturması.
    """
    try:
        # Appointment'u kontrol et
        appointment = db.query(Appointment).filter(
            Appointment.id == target_request.appointment_id
        ).first()
        
        if not appointment:
            raise HTTPException(
                status_code=404,
                detail="Randevu kaydı bulunamadı"
            )
        
        # Daha önce teşhis koyulmuş mu kontrol et
        existing_diagnosis = db.query(DoctorDiagnosis).filter(
            DoctorDiagnosis.doctor_id == current_user.id,
            DoctorDiagnosis.appointment_id == target_request.appointment_id
        ).first()
        
        if existing_diagnosis:
            # Mevcut teşhisi güncelle
            existing_diagnosis.doctor_target = target_request.target_value
            existing_diagnosis.doctor_notes = target_request.doctor_notes
            existing_diagnosis.updated_at = datetime.utcnow()
            db.commit()
            
            # CSV dosyasına kaydet (güncelleme için)
            diagnosis_data = {
                'age': existing_diagnosis.age,
                'gender': existing_diagnosis.gender,
                'height': existing_diagnosis.height,
                'weight': existing_diagnosis.weight,
                'ap_hi': existing_diagnosis.ap_hi,
                'ap_lo': existing_diagnosis.ap_lo,
                'cholesterol': existing_diagnosis.cholesterol,
                'gluc': existing_diagnosis.gluc,
                'smoke': existing_diagnosis.smoke,
                'alco': existing_diagnosis.alco,
                'active': existing_diagnosis.active,
                'doctor_target': existing_diagnosis.doctor_target
            }
            append_to_csv(diagnosis_data)
            
            # PredictionRecord'u silme işlemi kaldırıldı (foreign key constraint nedeniyle)
            # Sadece CSV'ye kaydetme işlemi yapılıyor
            
            return {
                "success": True,
                "message": "Hasta teşhisi güncellendi ve CSV dosyasına kaydedildi",
                "diagnosis_id": existing_diagnosis.id
            }
        else:
            # Yeni teşhis kaydı oluştur
            doctor_diagnosis = DoctorDiagnosis(
                appointment_id=target_request.appointment_id,
                doctor_id=current_user.id,
                age=appointment.age,
                gender=appointment.gender,
                height=appointment.height,
                weight=appointment.weight,
                ap_hi=appointment.ap_hi,
                ap_lo=appointment.ap_lo,
                cholesterol=appointment.cholesterol,
                gluc=appointment.gluc,
                smoke=appointment.smoke,
                alco=appointment.alco,
                active=appointment.active,
                risk_score=appointment.risk_score,
                priority_score=appointment.priority_score,
                prediction_label=appointment.prediction_label,
                prediction_probability=appointment.prediction_probability,
                model_name=appointment.model_name,
                doctor_target=target_request.target_value,
                doctor_notes=target_request.doctor_notes,
                diagnosis_status="completed"
            )
            
            db.add(doctor_diagnosis)
            db.commit()
            db.refresh(doctor_diagnosis)
            
            # CSV dosyasına kaydet (yeni kayıt için)
            diagnosis_data = {
                'age': doctor_diagnosis.age,
                'gender': doctor_diagnosis.gender,
                'height': doctor_diagnosis.height,
                'weight': doctor_diagnosis.weight,
                'ap_hi': doctor_diagnosis.ap_hi,
                'ap_lo': doctor_diagnosis.ap_lo,
                'cholesterol': doctor_diagnosis.cholesterol,
                'gluc': doctor_diagnosis.gluc,
                'smoke': doctor_diagnosis.smoke,
                'alco': doctor_diagnosis.alco,
                'active': doctor_diagnosis.active,
                'doctor_target': doctor_diagnosis.doctor_target
            }
            append_to_csv(diagnosis_data)
            
            # PredictionRecord'u silme işlemi kaldırıldı (foreign key constraint nedeniyle)
            # Sadece CSV'ye kaydetme işlemi yapılıyor
            
            return {
                "success": True,
                "message": "Hasta teşhisi kaydedildi ve CSV dosyasına eklendi",
                "diagnosis_id": doctor_diagnosis.id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Teşhis kaydetme hatası: {str(e)}"
        )

@router.get("/targets", response_model=Dict[str, Any])
async def get_doctor_targets(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Doktorun belirlediği tüm target değerlerini getir.
    """
    try:
        targets = db.query(DoctorTarget).filter(
            DoctorTarget.doctor_id == current_user.id
        ).order_by(DoctorTarget.created_at.desc()).all()
        
        target_list = []
        for target in targets:
            target_list.append({
                "id": target.id,
                "prediction_record_id": target.prediction_record_id,
                "patient_id": target.patient_id,
                "age": target.age,
                "gender": target.gender,
                "height": target.height,
                "weight": target.weight,
                "ap_hi": target.ap_hi,
                "ap_lo": target.ap_lo,
                "cholesterol": target.cholesterol,
                "gluc": target.gluc,
                "smoke": target.smoke,
                "alco": target.alco,
                "active": target.active,
                "target_value": target.target_value,
                "doctor_notes": target.doctor_notes,
                "created_at": target.created_at
            })
        
        return {
            "success": True,
            "message": f"{len(target_list)} target bulundu",
            "targets": target_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Target listesi getirilirken hata oluştu: {str(e)}"
        )

@router.get("/export-dataset", response_model=Dict[str, Any])
async def export_dataset(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Doktorun teşhis koyduğu tüm hastaların verilerini CSV formatında export et.
    """
    try:
        # Doktorun teşhis koyduğu tüm kayıtları getir
        diagnoses = db.query(DoctorDiagnosis).filter(
            DoctorDiagnosis.doctor_id == current_user.id
        ).order_by(DoctorDiagnosis.created_at.desc()).all()
        
        if not diagnoses:
            return {
                "success": False,
                "message": "Export edilecek teşhis kaydı bulunamadı",
                "data": []
            }
        
        # CSV formatında veri hazırla (cardio_train.csv formatına uygun)
        csv_data = []
        for diagnosis in diagnoses:
            csv_data.append({
                "id": diagnosis.id,
                "age": diagnosis.age,
                "gender": diagnosis.gender,
                "height": diagnosis.height,
                "weight": diagnosis.weight,
                "ap_hi": diagnosis.ap_hi,
                "ap_lo": diagnosis.ap_lo,
                "cholesterol": diagnosis.cholesterol,
                "gluc": diagnosis.gluc,
                "smoke": diagnosis.smoke,
                "alco": diagnosis.alco,
                "active": diagnosis.active,
                "cardio": diagnosis.doctor_target  # doctor_target -> cardio olarak değiştir
            })
        
        return {
            "success": True,
            "message": f"{len(csv_data)} teşhis kaydı export edildi",
            "data": csv_data,
            "total_records": len(csv_data)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset export hatası: {str(e)}"
        )

@router.get("/diagnoses", response_model=Dict[str, Any])
async def get_doctor_diagnoses(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Doktorun teşhis koyduğu tüm hastaları getir.
    """
    try:
        diagnoses = db.query(DoctorDiagnosis).filter(
            DoctorDiagnosis.doctor_id == current_user.id
        ).order_by(DoctorDiagnosis.created_at.desc()).all()
        
        diagnosis_list = []
        for diagnosis in diagnoses:
            diagnosis_list.append({
                "id": diagnosis.id,
                "appointment_id": diagnosis.appointment_id,
                "age": diagnosis.age,
                "gender": diagnosis.gender,
                "height": diagnosis.height,
                "weight": diagnosis.weight,
                "ap_hi": diagnosis.ap_hi,
                "ap_lo": diagnosis.ap_lo,
                "cholesterol": diagnosis.cholesterol,
                "gluc": diagnosis.gluc,
                "smoke": diagnosis.smoke,
                "alco": diagnosis.alco,
                "active": diagnosis.active,
                "risk_score": diagnosis.risk_score,
                "priority_score": diagnosis.priority_score,
                "prediction_label": diagnosis.prediction_label,
                "prediction_probability": diagnosis.prediction_probability,
                "model_name": diagnosis.model_name,
                "doctor_target": diagnosis.doctor_target,
                "doctor_notes": diagnosis.doctor_notes,
                "diagnosis_status": diagnosis.diagnosis_status,
                "created_at": diagnosis.created_at.isoformat()
            })
        
        return {
            "success": True,
            "message": f"{len(diagnosis_list)} teşhis kaydı bulundu",
            "diagnoses": diagnosis_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Teşhis listesi getirilirken hata oluştu: {str(e)}"
        )

@router.post("/assign-appointments", response_model=Dict[str, Any])
async def assign_appointments(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_doctor)
):
    """
    Manuel olarak randevu atama işlemini gerçekleştir.
    """
    try:
        # Mevcut randevu kayıtlarını temizle
        deleted_count = db.query(Appointment).count()
        db.query(Appointment).delete()
        db.commit()
        
        # En yüksek risk skoruna sahip 20 hastayı getir
        top_patients = db.query(PredictionRecord).order_by(
            PredictionRecord.risk_score.desc(),
            PredictionRecord.priority_score.desc(),
            PredictionRecord.created_at.desc()
        ).limit(20).all()
        
        if len(top_patients) == 0:
            return {
                "success": False,
                "message": "Hiç hasta bulunamadı!",
                "data": {}
            }
        
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
        
        db.commit()
        
        return {
            "success": True,
            "message": f"{len(top_patients)} hasta başarıyla randevu tablosuna atandı",
            "data": {
                "deleted_count": deleted_count,
                "assigned_count": len(top_patients)
            }
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Randevu atama hatası: {str(e)}"
        )

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
