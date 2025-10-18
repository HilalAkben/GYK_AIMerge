import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal
from sqlalchemy import text

def clean_prediction_records():
    """
    appointments tablosunda olan ve prediction_records tablosunda da bulunan 
    kayıtları sadece prediction_records tablosundan sil.
    appointments tablosundan silme!
    """
    db = SessionLocal()
    try:
        print("Prediction records temizleme işlemi başlatılıyor...")
        
        # 1. Önce mevcut durumu kontrol et
        appointments_count = db.execute(text("SELECT COUNT(*) FROM appointments")).fetchone()[0]
        prediction_count_before = db.execute(text("SELECT COUNT(*) FROM prediction_records")).fetchone()[0]
        
        print(f"Appointments tablosunda {appointments_count} kayıt var")
        print(f"Prediction_records tablosunda {prediction_count_before} kayıt var")
        
        # 2. appointments tablosunda olan prediction_record_id'leri bul
        appointments_ids = db.execute(text("""
            SELECT DISTINCT prediction_record_id 
            FROM appointments 
            WHERE prediction_record_id IS NOT NULL
        """)).fetchall()
        
        appointment_prediction_ids = [row[0] for row in appointments_ids]
        print(f"Appointments tablosunda {len(appointment_prediction_ids)} farklı prediction_record_id var")
        
        if not appointment_prediction_ids:
            print("Silinecek prediction record bulunamadı")
            return
        
        # 3. Önce appointments tablosundaki prediction_record_id'leri NULL yap
        # (Foreign key constraint nedeniyle)
        print("Appointments tablosundaki prediction_record_id'ler NULL yapiliyor...")
        appointments_updated = db.execute(text("""
            UPDATE appointments 
            SET prediction_record_id = NULL 
            WHERE prediction_record_id IS NOT NULL
        """)).rowcount
        
        print(f"Appointments tablosunda {appointments_updated} kayit guncellendi")
        
        # 4. Şimdi prediction_records'ları sil
        print("Prediction records siliniyor...")
        deleted_count = 0
        for pred_id in appointment_prediction_ids:
            result = db.execute(text("DELETE FROM prediction_records WHERE id = :pred_id"), {"pred_id": pred_id})
            deleted_count += result.rowcount
        
        db.commit()
        
        # 4. Son durumu kontrol et
        prediction_count_after = db.execute(text("SELECT COUNT(*) FROM prediction_records")).fetchone()[0]
        appointments_count_after = db.execute(text("SELECT COUNT(*) FROM appointments")).fetchone()[0]
        
        print(f"\nSUCCESS: Islem tamamlandi!")
        print(f"Silinen prediction record sayisi: {deleted_count}")
        print(f"Prediction_records tablosu: {prediction_count_before} -> {prediction_count_after}")
        print(f"Appointments tablosu: {appointments_count} -> {appointments_count_after} (degismedi)")
        
        # 5. Kalan prediction records'ları göster
        remaining_predictions = db.execute(text("""
            SELECT id, risk_score, prediction, probability, model_id
            FROM prediction_records 
            ORDER BY risk_score DESC 
            LIMIT 5
        """)).fetchall()
        
        print(f"\nKalan ilk 5 prediction record:")
        for pred in remaining_predictions:
            print(f"  - ID: {pred[0]}, Risk: {pred[1]:.3f}, Tahmin: {pred[2]}, Olasilik: {pred[3]:.3f}, Model: {pred[4]}")
            
    except Exception as e:
        print(f"ERROR: Hata olustu: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    clean_prediction_records()
