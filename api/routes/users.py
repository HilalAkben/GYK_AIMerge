"""
Kullanıcı Profil API Routes
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from api.core.database import get_db, User, Appointment, PredictionRecord, DoctorDiagnosis
from api.services.auth_service import get_current_active_user
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool

class DoctorInfo(BaseModel):
    id: int
    full_name: str
    username: str
    email: str

class AppointmentHistoryResponse(BaseModel):
    id: int
    date: str
    time: str
    status: str
    riskLevel: str
    priority: str
    model: str
    risk_score: float
    prediction_probability: float
    prediction_label: int
    created_at: str
    appointment_date: Optional[str] = None
    assigned_at: Optional[str] = None
    queue_position: Optional[int] = None  # Randevu sırası eklendi
    doctor: Optional[DoctorInfo] = None
    doctor_notes: Optional[str] = None
    doctor_target: Optional[int] = None
    diagnosis_status: Optional[str] = None

class UserAppointmentsResponse(BaseModel):
    success: bool
    message: str
    appointments: List[AppointmentHistoryResponse]
    total_count: int

@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Kullanıcı profil bilgilerini getir."""
    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active
    )

@router.get("/{user_id}/appointments", response_model=UserAppointmentsResponse)
async def get_user_appointments(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Kullanıcının tüm randevu kayıtlarını getir."""
    
    # Kullanıcının kendi randevularını görüntülemesini kontrol et
    if current_user.id != user_id and current_user.role not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=403, 
            detail="Bu kullanıcının randevularını görüntüleme yetkiniz yok"
        )
    
    try:
        # Kullanıcının tüm randevularını getir (appointments tablosundan)
        # Doktor bilgileri ve teşhis bilgileri ile birlikte
        appointments = db.query(Appointment).options(
            joinedload(Appointment.doctor_diagnoses).joinedload(DoctorDiagnosis.doctor)
        ).filter(
            Appointment.patient_id == user_id
        ).order_by(desc(Appointment.created_at)).all()
        
        # Prediction record'ları da getir (eğer appointment yoksa)
        prediction_records = db.query(PredictionRecord).filter(
            PredictionRecord.patient_external_id == user_id
        ).order_by(desc(PredictionRecord.created_at)).all()
        
        appointment_list = []
        
        # Appointment'ları işle
        for apt in appointments:
            risk_level = "Yüksek" if apt.prediction_label == 1 else "Düşük"
            priority_map = {
                "low": "Düşük",
                "normal": "Normal", 
                "high": "Yüksek",
                "urgent": "Acil"
            }
            
            # Doktor bilgilerini al
            doctor_info = None
            doctor_notes = None
            doctor_target = None
            diagnosis_status = None
            
            if apt.doctor_diagnoses:
                # En son teşhisi al
                latest_diagnosis = apt.doctor_diagnoses[0]
                if latest_diagnosis.doctor:
                    doctor_info = DoctorInfo(
                        id=latest_diagnosis.doctor.id,
                        full_name=latest_diagnosis.doctor.full_name,
                        username=latest_diagnosis.doctor.username,
                        email=latest_diagnosis.doctor.email
                    )
                doctor_notes = latest_diagnosis.doctor_notes
                doctor_target = latest_diagnosis.doctor_target
                diagnosis_status = latest_diagnosis.diagnosis_status
            
            # Status'u belirle: 
            # - appointment_date varsa "tamamlandı"
            # - queue_position varsa "sırada" 
            # - yoksa "beklemede"
            if apt.appointment_date:
                appointment_status = "tamamlandı"
            elif apt.queue_position:
                appointment_status = "sırada"
            else:
                appointment_status = "beklemede"
            
            appointment_list.append(AppointmentHistoryResponse(
                id=apt.id,
                date=apt.created_at.strftime("%d.%m.%Y"),
                time=apt.created_at.strftime("%H:%M"),
                status=appointment_status,
                riskLevel=risk_level,
                priority=priority_map.get(apt.priority, "Normal"),
                model=apt.model_name,
                risk_score=apt.risk_score,
                prediction_probability=apt.prediction_probability,
                prediction_label=apt.prediction_label,
                created_at=apt.created_at.isoformat(),
                appointment_date=apt.appointment_date.isoformat() if apt.appointment_date else None,
                assigned_at=apt.assigned_at.isoformat() if apt.assigned_at else None,
                queue_position=apt.queue_position,  # Randevu sırası eklendi
                doctor=doctor_info,
                doctor_notes=doctor_notes,
                doctor_target=doctor_target,
                diagnosis_status=diagnosis_status
            ))
        
        # Prediction record'ları işle (appointment olmayanlar)
        existing_prediction_ids = {apt.prediction_record_id for apt in appointments}
        
        for pred in prediction_records:
            if pred.id not in existing_prediction_ids:
                risk_level = "Yüksek" if pred.prediction == 1 else "Düşük"
                
                appointment_list.append(AppointmentHistoryResponse(
                    id=pred.id,
                    date=pred.created_at.strftime("%d.%m.%Y"),
                    time=pred.created_at.strftime("%H:%M"),
                    status="beklemede",  # PredictionRecord'lar beklemede olmalı
                    riskLevel=risk_level,
                    priority="Normal",
                    model=f"Model_{pred.model_id}",
                    risk_score=pred.risk_score,
                    prediction_probability=pred.probability,
                    prediction_label=pred.prediction,
                    created_at=pred.created_at.isoformat(),
                    appointment_date=None,
                    assigned_at=None,
                    doctor=None,
                    doctor_notes=None,
                    doctor_target=None,
                    diagnosis_status=None
                ))
        
        # Tarihe göre sırala (en yeni önce)
        appointment_list.sort(key=lambda x: x.created_at, reverse=True)
        
        return UserAppointmentsResponse(
            success=True,
            message=f"Kullanıcının {len(appointment_list)} randevu kaydı bulundu",
            appointments=appointment_list,
            total_count=len(appointment_list)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Randevu kayıtları getirilirken hata oluştu: {str(e)}"
        )
