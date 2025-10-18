#!/usr/bin/env python3
"""
PostgreSQL için otomatik randevu atama sistemi
MySQL EVENT benzeri işlevsellik
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal
from sqlalchemy import text
from datetime import datetime, timedelta
import time
import threading
import json

class PostgresAppointmentSystem:
    def __init__(self):
        self.is_running = False
        self.thread = None
        
    def start(self):
        """Sistemi başlat - her 10 dakikada bir çalışacak"""
        if self.is_running:
            print("⚠️ Sistem zaten çalışıyor!")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._run_periodic_assignment)
        self.thread.daemon = True
        self.thread.start()
        print("🔄 PostgreSQL randevu sistemi başlatıldı (her 10 dakikada bir)")
        
    def stop(self):
        """Sistemi durdur"""
        self.is_running = False
        if self.thread:
            self.thread.join()
        print("⏹️ PostgreSQL randevu sistemi durduruldu")
        
    def _run_periodic_assignment(self):
        """Her 10 dakikada bir randevu atama işlemini çalıştır"""
        while self.is_running:
            try:
                self.assign_appointments()
                print(f"✅ {datetime.now().strftime('%H:%M:%S')} - Randevu atama işlemi tamamlandı")
            except Exception as e:
                print(f"❌ {datetime.now().strftime('%H:%M:%S')} - Randevu atama hatası: {e}")
            
            # 10 dakika bekle (600 saniye)
            for _ in range(600):  # 600 saniye = 10 dakika
                if not self.is_running:
                    break
                time.sleep(1)
    
    def assign_appointments(self):
        """Randevu atama işlemi - MySQL EVENT'in PostgreSQL karşılığı"""
        db = SessionLocal()
        try:
            print("Randevu atama islemi baslatiliyor...")
            
            # 1. Mevcut randevuları temizle
            db.execute(text("DELETE FROM appointments"))
            db.commit()
            print("Eski randevular temizlendi")
            
            # 2. En yüksek riskli 20 hastayı seç ve JSON'dan hasta bilgilerini çıkar
            result = db.execute(text("""
                SELECT id, patient_data, risk_score, prediction, probability, model_id, patient_external_id
                FROM prediction_records 
                ORDER BY risk_score DESC 
                LIMIT 20
            """))
            
            appointments_data = []
            for i, row in enumerate(result, 1):
                try:
                    # JSON'dan hasta bilgilerini çıkar
                    patient_data = json.loads(row.patient_data)
                    
                    # Öncelik seviyesini belirle
                    if row.risk_score >= 0.8:
                        priority = 'urgent'
                    elif row.risk_score >= 0.6:
                        priority = 'high'
                    elif row.risk_score >= 0.4:
                        priority = 'normal'
                    else:
                        priority = 'low'
                    
                    # patient_id'yi bul (patient_external_id'den)
                    patient_id = None
                    if row.patient_external_id:
                        # patient_external_id'yi users tablosunda ara
                        user_result = db.execute(text("""
                            SELECT id FROM users WHERE id = :patient_external_id
                        """), {"patient_external_id": row.patient_external_id})
                        user_row = user_result.fetchone()
                        if user_row:
                            patient_id = user_row[0]
                    
                    appointment_data = {
                        'prediction_record_id': row.id,
                        'patient_id': patient_id,  # Hasta ID'si (varsa)
                        'age': patient_data.get('age', 0),
                        'gender': patient_data.get('gender', 1),
                        'height': float(patient_data.get('height', 0.0)),
                        'weight': float(patient_data.get('weight', 0.0)),
                        'ap_hi': patient_data.get('ap_hi', 0),
                        'ap_lo': patient_data.get('ap_lo', 0),
                        'cholesterol': patient_data.get('cholesterol', 1),
                        'gluc': patient_data.get('gluc', 1),
                        'smoke': 1 if patient_data.get('smoke', 0) else 0,
                        'alco': 1 if patient_data.get('alco', 0) else 0,
                        'active': 1 if patient_data.get('active', 0) else 0,
                        'risk_score': float(row.risk_score),
                        'priority_score': float(row.risk_score),
                        'prediction_label': row.prediction,
                        'prediction_probability': float(row.probability),
                        'model_name': f'Model_{row.model_id}',
                        'status': 'assigned',
                        'priority': priority,
                        'queue_position': i,
                        'appointment_date': datetime.utcnow() + timedelta(days=1),
                        'assigned_at': datetime.utcnow(),
                        'created_at': datetime.utcnow(),
                        'updated_at': datetime.utcnow()
                    }
                    appointments_data.append(appointment_data)
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parse hatasi (ID: {row.id}): {e}")
                    continue
                except Exception as e:
                    print(f"Hasta verisi isleme hatasi (ID: {row.id}): {e}")
                    continue
            
            # 3. Appointments tablosuna ekle
            for data in appointments_data:
                db.execute(text("""
                    INSERT INTO appointments (
                        prediction_record_id, patient_id, age, gender, height, weight, ap_hi, ap_lo,
                        cholesterol, gluc, smoke, alco, active, risk_score, priority_score,
                        prediction_label, prediction_probability, model_name, status, priority,
                        queue_position, appointment_date, assigned_at, created_at, updated_at
                    ) VALUES (
                        :prediction_record_id, :patient_id, :age, :gender, :height, :weight, :ap_hi, :ap_lo,
                        :cholesterol, :gluc, :smoke, :alco, :active, :risk_score, :priority_score,
                        :prediction_label, :prediction_probability, :model_name, :status, :priority,
                        :queue_position, :appointment_date, :assigned_at, :created_at, :updated_at
                    )
                """), data)
            
            db.commit()
            print(f"{len(appointments_data)} hasta randevu tablosuna atandi")
            
            # 4. Sonuçları göster
            result = db.execute(text("""
                SELECT 
                    id, age, gender, risk_score, priority, queue_position, 
                    prediction_label, model_name, status
                FROM appointments 
                ORDER BY queue_position 
                LIMIT 5
            """))
            appointments = result.fetchall()
            
            print("Ilk 5 randevu:")
            for app in appointments:
                print(f"  - ID: {app[0]}, Yas: {app[1]}, Cinsiyet: {app[2]}, Risk: {app[3]:.3f}, Oncelik: {app[4]}, Sira: {app[5]}, Tahmin: {app[6]}, Model: {app[7]}, Durum: {app[8]}")
                
        except Exception as e:
            print(f"Randevu atama hatasi: {e}")
            db.rollback()
        finally:
            db.close()
    
    def manual_assign(self):
        """Manuel randevu atama (test için)"""
        print("Manuel randevu atama baslatiliyor...")
        self.assign_appointments()

# Global instance
appointment_system = PostgresAppointmentSystem()

def start_appointment_system():
    """Sistemi başlat"""
    appointment_system.start()

def stop_appointment_system():
    """Sistemi durdur"""
    appointment_system.stop()

def manual_assign_appointments():
    """Manuel randevu atama"""
    appointment_system.manual_assign()

if __name__ == "__main__":
    print("🚀 PostgreSQL Randevu Sistemi")
    print("1. Manuel randevu atama")
    print("2. Otomatik sistem başlat")
    print("3. Sistemi durdur")
    
    choice = input("Seçiminiz (1-3): ")
    
    if choice == "1":
        manual_assign_appointments()
    elif choice == "2":
        start_appointment_system()
        print("Sistem çalışıyor... Ctrl+C ile durdurun")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_appointment_system()
    elif choice == "3":
        stop_appointment_system()
    else:
        print("Geçersiz seçim!")
