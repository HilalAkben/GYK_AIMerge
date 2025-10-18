from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any
from datetime import datetime
from api.core.database import get_db
from api.core.database import PredictionRecord, ModelRecord, Appointment, User
from api.services.auth_service import require_admin, get_current_user
from api.services.scheduler_service import scheduler_service

router = APIRouter()

@router.get("/appointment-queue", response_model=Dict[str, Any])
async def get_appointment_queue(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Randevu sırasındaki hasta kayıtlarını getir.
    Sadece admin kullanıcılar erişebilir.
    """
    try:
        # Tüm tahmin kayıtlarını getir
        records = db.query(PredictionRecord).order_by(
            PredictionRecord.risk_score.desc(),
            PredictionRecord.created_at.desc()
        ).all()
        
        # İstatistikler
        total_records = len(records)
        high_priority = len([r for r in records if r.risk_score >= 0.7])
        medium_priority = len([r for r in records if 0.4 <= r.risk_score < 0.7])
        low_priority = len([r for r in records if r.risk_score < 0.4])
        
        # Model isimlerini çek
        model_names = {}
        model_records = db.query(ModelRecord).filter(ModelRecord.is_active == True).all()
        print(f"DEBUG: Bulunan modeller: {[(m.id, m.model_name) for m in model_records]}")
        for model in model_records:
            model_names[model.id] = model.model_name
        
        # Kayıtları formatla
        formatted_records = []
        for record in records:
            # Model ismini model_names'den çek, yoksa fallback olarak model_id kullan
            if record.model_id in model_names:
                model_name = model_names[record.model_id]
            else:
                # Fallback: Model ID'sine göre anlamlı isim ver
                model_fallback_names = {
                    1: "CatBoost Model",
                    2: "XGBoost Model", 
                    3: "Random Forest Model",
                    4: "Logistic Regression Model",
                    5: "LightGBM Model"
                }
                model_name = model_fallback_names.get(record.model_id, f"Model_{record.model_id}")
            
            formatted_records.append({
                "id": record.id,
                "user_id": None,  # PredictionRecord'da user_id yok
                "model_name": model_name,  # Gerçek model ismi
                "age": record.age,
                "gender": record.gender,
                "height": float(record.height) if record.height else None,
                "weight": float(record.weight) if record.weight else None,
                "ap_hi": record.ap_hi,
                "ap_lo": record.ap_lo,
                "cholesterol": record.cholesterol,
                "gluc": record.gluc,
                "smoke": record.smoke,
                "alco": record.alco,
                "active": record.active,
                "risk_score": float(record.risk_score),
                "weighted_factors_score": 0.0,  # Bu alan yok, varsayılan değer
                "priority_score": float(record.risk_score),  # risk_score'u priority olarak kullan
                "prediction_probability": float(record.probability),  # probability alanı var
                "prediction_label": int(record.prediction),  # prediction alanı var
                "created_at": record.created_at.isoformat() if record.created_at else None,
            })
        
        return {
            "success": True,
            "message": f"Randevu sırası başarıyla getirildi. Toplam {total_records} kayıt bulundu.",
            "data": {
                "records": formatted_records,
                "statistics": {
                    "total": total_records,
                    "high_priority": high_priority,
                    "medium_priority": medium_priority,
                    "low_priority": low_priority,
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Randevu sırası getirilirken hata oluştu: {str(e)}"
        )

@router.get("/appointment-queue/status/{prediction_id}", response_model=Dict[str, Any])
async def get_queue_status(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Belirli bir tahmin kaydının sıra konumunu döndür.
    """
    try:
        ordered = db.query(PredictionRecord).order_by(
            PredictionRecord.risk_score.desc(),
            PredictionRecord.created_at.desc()
        ).all()

        ids = [r.id for r in ordered]
        if prediction_id not in ids:
            return {
                "success": True,
                "message": "Kayıt bulunamadı, beklemede.",
                "data": {
                    "status": "waiting",
                    "position": None,
                    "total": len(ids)
                }
            }

        position = ids.index(prediction_id) + 1
        record = next(r for r in ordered if r.id == prediction_id)

        return {
            "success": True,
            "message": "Sıra bilgisi başarıyla getirildi.",
            "data": {
                "status": "queued",
                "position": position,
                "total": len(ids),
                "record": {
                    "id": record.id,
                    "model_id": record.model_id,
                    "age": record.age,
                    "gender": record.gender,
                    "risk_score": float(record.risk_score),
                    "probability": float(record.probability),
                    "prediction": int(record.prediction),
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sıra bilgisi alınırken hata: {str(e)}")


@router.get("/scheduler/status", response_model=Dict[str, Any])
async def get_scheduler_status(
    current_user = Depends(require_admin)
):
    """
    Scheduler durumunu getir.
    Admin yetkisi gereklidir.
    """
    try:
        status = scheduler_service.get_scheduler_status()
        return {
            "success": True,
            "message": "Scheduler durumu başarıyla getirildi",
            "data": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scheduler durumu alınırken hata: {str(e)}")


@router.post("/scheduler/manual-sort", response_model=Dict[str, Any])
async def manual_sorting(
    current_user = Depends(require_admin)
):
    """
    Manuel sıralama yap.
    Admin yetkisi gereklidir.
    """
    try:
        result = await scheduler_service.manual_sorting()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Manuel sıralama hatası: {str(e)}")


@router.post("/priority-sort", response_model=Dict[str, Any])
async def priority_sort_appointments(
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Prediction records tablosundaki hastaları öncelik sırasına göre sıralayıp
    ilk 20 hastayı appointments tablosuna kaydeder ve prediction_records tablosundan siler.
    Sadece admin kullanıcılar erişebilir.
    """
    try:
        # Mevcut appointments tablosundaki kayıt sayısını al
        deleted_appointments = db.query(Appointment).count()
        
        # Mevcut appointments'ları temizle (doctor_diagnoses ile birlikte)
        from api.core.database import DoctorDiagnosis
        
        # Önce doctor_diagnoses tablosundaki ilgili kayıtları sil
        db.query(DoctorDiagnosis).delete()
        
        # Sonra appointments tablosunu temizle
        db.query(Appointment).delete()
        db.commit()
        
        print(f"DEBUG: {deleted_appointments} eski appointment kaydı temizlendi")
        
        # Prediction records tablosundan en yüksek risk skoruna sahip 20 hastayı getir
        top_patients = db.query(PredictionRecord).order_by(
            PredictionRecord.risk_score.desc(),
            PredictionRecord.created_at.desc()
        ).limit(20).all()
        
        if len(top_patients) == 0:
            return {
                "success": False,
                "message": "Hiç hasta bulunamadı!",
                "data": {
                    "appointments_created": 0,
                    "patients_deleted": 0,
                    "deleted_appointments": deleted_appointments
                }
            }
        
        # Model isimlerini çek
        model_names = {}
        model_records = db.query(ModelRecord).filter(ModelRecord.is_active == True).all()
        for model in model_records:
            model_names[model.id] = model.model_name
        
        # Her hasta için randevu kaydı oluştur
        created_appointments = []
        for i, patient in enumerate(top_patients, 1):
            # Model ismini belirle
            if patient.model_id in model_names:
                model_name = model_names[patient.model_id]
            else:
                model_fallback_names = {
                    1: "CatBoost Model",
                    2: "XGBoost Model", 
                    3: "Random Forest Model",
                    4: "Logistic Regression Model",
                    5: "LightGBM Model"
                }
                model_name = model_fallback_names.get(patient.model_id, f"Model_{patient.model_id}")
            
            # Öncelik belirle
            def determine_priority(risk_score: float) -> str:
                if risk_score >= 0.8:
                    return "urgent"
                elif risk_score >= 0.6:
                    return "high"
                elif risk_score >= 0.4:
                    return "normal"
                else:
                    return "low"
            
            appointment = Appointment(
                prediction_record_id=patient.id,
                patient_id=patient.patient_external_id,
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
                priority_score=patient.risk_score,
                prediction_label=patient.prediction,
                prediction_probability=patient.probability,
                model_name=model_name,
                queue_position=i,
                status="pending",
                priority=determine_priority(patient.risk_score),
                assigned_at=datetime.utcnow()
            )
            db.add(appointment)
            created_appointments.append(patient.id)
        
        # Appointments tablosuna kaydet
        db.commit()
        
        # Kaydedilen hastaları prediction_records tablosundan sil
        # Foreign key constraint nedeniyle silme işlemi yapmıyoruz
        # Bunun yerine sadece işaretliyoruz
        processed_patients = len(created_appointments)
        
        # Not: Foreign key constraint nedeniyle prediction_records silinemiyor
        # Bu yüzden sadece appointments tablosuna kaydediyoruz
        # prediction_records tablosundaki kayıtlar kalacak
        
        # Appointments tablosundan yeni eklenen kayıtları getir
        # Eğer created_appointments boş değilse ve prediction_record_id'ler mevcutsa
        if created_appointments:
            new_appointments = db.query(Appointment).filter(
                Appointment.prediction_record_id.in_(created_appointments)
            ).order_by(Appointment.queue_position.asc()).all()
        else:
            # Eğer created_appointments boşsa, en son eklenen appointments'ları getir
            new_appointments = db.query(Appointment).order_by(
                Appointment.created_at.desc()
            ).limit(20).all()
        
        return {
            "success": True,
            "message": f"Öncelik sıralaması başarıyla tamamlandı! {len(created_appointments)} hasta randevu tablosuna eklendi.",
            "data": {
                "appointments_created": len(created_appointments),
                "patients_processed": processed_patients,
                "deleted_appointments": deleted_appointments,
                "top_patients": [
                    {
                        "id": apt.id,
                        "age": apt.age,
                        "gender": apt.gender,
                        "height": float(apt.height) if apt.height else None,
                        "weight": float(apt.weight) if apt.weight else None,
                        "ap_hi": apt.ap_hi,
                        "ap_lo": apt.ap_lo,
                        "cholesterol": apt.cholesterol,
                        "gluc": apt.gluc,
                        "smoke": apt.smoke,
                        "alco": apt.alco,
                        "active": apt.active,
                        "risk_score": float(apt.risk_score),
                        "prediction": int(apt.prediction_label),
                        "prediction_probability": float(apt.prediction_probability),
                        "model_name": apt.model_name,
                        "queue_position": apt.queue_position,
                        "priority": apt.priority,
                        "status": apt.status,
                        "created_at": apt.created_at.isoformat() if apt.created_at else None
                    } for apt in new_appointments
                ]
            }
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Öncelik sıralaması yapılırken hata oluştu: {str(e)}"
        )


@router.post("/assign-doctor/{appointment_id}", response_model=Dict[str, Any])
async def assign_doctor_to_appointment(
    appointment_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Belirli bir randevuya doktor atar.
    Sadece admin kullanıcılar erişebilir.
    """
    try:
        print(f"DEBUG: Doktor atama isteği - appointment_id: {appointment_id}")
        
        # Randevuyu bul
        appointment = db.query(Appointment).filter(Appointment.id == appointment_id).first()
        print(f"DEBUG: Bulunan appointment: {appointment}")
        
        if not appointment:
            print(f"DEBUG: Appointment {appointment_id} bulunamadı!")
            raise HTTPException(
                status_code=404,
                detail="Randevu bulunamadı"
            )
        
        # Role'ü 'doctor' olan kullanıcıyı bul
        doctor = db.query(User).filter(User.role == "doctor").first()
        print(f"DEBUG: Bulunan doktor: {doctor}")
        
        if not doctor:
            print("DEBUG: Doktor bulunamadı!")
            raise HTTPException(
                status_code=404,
                detail="Sistemde doktor bulunamadı"
            )
        
        # Randevuya doktor ata
        print(f"DEBUG: Appointment {appointment_id} güncelleniyor - doctor_id: {doctor.id}")
        appointment.doctor_id = doctor.id
        appointment.assigned_at = datetime.utcnow()
        appointment.status = "assigned"
        print(f"DEBUG: Appointment güncellendi - doctor_id: {appointment.doctor_id}, status: {appointment.status}")
        
        # DoctorDiagnosis tablosuna da kayıt ekle
        from api.core.database import DoctorDiagnosis
        doctor_diagnosis = DoctorDiagnosis(
            appointment_id=appointment.id,
            doctor_id=doctor.id,
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
            doctor_target=0,  # Varsayılan değer, doktor daha sonra güncelleyecek
            doctor_notes="Doktor atandı, teşhis bekleniyor",
            diagnosis_status="pending"
        )
        db.add(doctor_diagnosis)
        
        print(f"DEBUG: Veritabanına commit ediliyor...")
        db.commit()
        print(f"DEBUG: Commit başarılı!")
        
        return {
            "success": True,
            "message": f"Randevu #{appointment_id} başarıyla Dr. {doctor.username}'a atandı",
            "data": {
                "appointment_id": appointment_id,
                "doctor_id": doctor.id,
                "doctor_name": doctor.username,
                "assigned_at": appointment.assigned_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Doktor atama işlemi sırasında hata oluştu: {str(e)}"
        )
