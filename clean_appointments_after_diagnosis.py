import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal
from sqlalchemy import text

def clean_appointments_after_diagnosis():
    """
    appointments tablosundaki kayıtları sadece doctor_target verildikten sonra sil.
    doctor_target verilmemiş kayıtları silme!
    """
    db = SessionLocal()
    try:
        print("Appointments temizleme işlemi başlatılıyor...")
        
        # 1. Önce mevcut durumu kontrol et
        appointments_count = db.execute(text("SELECT COUNT(*) FROM appointments")).fetchone()[0]
        doctor_diagnoses_count = db.execute(text("SELECT COUNT(*) FROM doctor_diagnoses")).fetchone()[0]
        
        print(f"Appointments tablosunda {appointments_count} kayıt var")
        print(f"Doctor_diagnoses tablosunda {doctor_diagnoses_count} kayıt var")
        
        # 2. Doctor_target verilmiş (doctor_diagnoses tablosunda olan) appointment_id'leri bul
        diagnosed_appointments = db.execute(text("""
            SELECT DISTINCT appointment_id 
            FROM doctor_diagnoses 
            WHERE appointment_id IS NOT NULL
        """)).fetchall()
        
        diagnosed_appointment_ids = [row[0] for row in diagnosed_appointments]
        print(f"Doctor_target verilmiş {len(diagnosed_appointment_ids)} appointment var")
        
        if not diagnosed_appointment_ids:
            print("Silinecek appointment bulunamadı (hiçbirine doctor_target verilmemiş)")
            return
        
        # 3. Önce doctor_diagnoses tablosundaki appointment_id'leri NULL yap
        print("Doctor_diagnoses tablosundaki appointment_id'ler NULL yapiliyor...")
        doctor_diagnoses_updated = 0
        for appointment_id in diagnosed_appointment_ids:
            result = db.execute(text("UPDATE doctor_diagnoses SET appointment_id = NULL WHERE appointment_id = :appointment_id"), {"appointment_id": appointment_id})
            doctor_diagnoses_updated += result.rowcount
        
        print(f"Doctor_diagnoses tablosunda {doctor_diagnoses_updated} kayit guncellendi")
        
        # 4. Şimdi appointments'ları sil (doctor_diagnoses tablosu kalıcı kalacak)
        print("Doctor_target verilmiş appointments siliniyor...")
        deleted_count = 0
        for appointment_id in diagnosed_appointment_ids:
            result = db.execute(text("DELETE FROM appointments WHERE id = :appointment_id"), {"appointment_id": appointment_id})
            deleted_count += result.rowcount
        
        db.commit()
        
        # 4. Son durumu kontrol et
        appointments_count_after = db.execute(text("SELECT COUNT(*) FROM appointments")).fetchone()[0]
        doctor_diagnoses_count_after = db.execute(text("SELECT COUNT(*) FROM doctor_diagnoses")).fetchone()[0]
        
        print(f"\nSUCCESS: Islem tamamlandi!")
        print(f"Silinen appointment sayisi: {deleted_count}")
        print(f"Appointments tablosu: {appointments_count} -> {appointments_count_after}")
        print(f"Doctor_diagnoses tablosu: {doctor_diagnoses_count} -> {doctor_diagnoses_count_after} (degismedi)")
        
        # 5. Kalan appointments'ları göster
        remaining_appointments = db.execute(text("""
            SELECT id, age, risk_score, priority, status, patient_id
            FROM appointments 
            ORDER BY risk_score DESC 
            LIMIT 5
        """)).fetchall()
        
        print(f"\nKalan ilk 5 appointment:")
        for apt in remaining_appointments:
            print(f"  - ID: {apt[0]}, Yaş: {apt[1]}, Risk: {apt[2]:.3f}, Öncelik: {apt[3]}, Durum: {apt[4]}, Patient ID: {apt[5]}")
            
    except Exception as e:
        print(f"ERROR: Hata olustu: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    clean_appointments_after_diagnosis()
