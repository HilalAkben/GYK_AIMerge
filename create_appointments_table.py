#!/usr/bin/env python3
"""
Appointments tablosunu oluşturmak için script
"""

import sqlite3
from datetime import datetime

def create_appointments_table():
    try:
        # Veritabanına bağlan
        conn = sqlite3.connect('cardio_models.db')
        cursor = conn.cursor()
        
        # Appointments tablosunu oluştur
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            prediction_record_id INTEGER UNIQUE,
            age INTEGER NOT NULL,
            gender INTEGER NOT NULL,
            height REAL,
            weight REAL,
            ap_hi INTEGER NOT NULL,
            ap_lo INTEGER NOT NULL,
            cholesterol INTEGER NOT NULL,
            gluc INTEGER NOT NULL,
            smoke INTEGER NOT NULL,
            alco INTEGER NOT NULL,
            active INTEGER NOT NULL,
            risk_score REAL NOT NULL,
            priority_score REAL NOT NULL,
            prediction_label INTEGER NOT NULL,
            prediction_probability REAL NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            priority VARCHAR(20) DEFAULT 'normal',
            queue_position INTEGER,
            appointment_date DATETIME,
            assigned_at DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES users (id)
        )
        ''')
        
        # Users tablosunu oluştur
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            full_name VARCHAR(100) NOT NULL,
            hashed_password VARCHAR(255) NOT NULL,
            role VARCHAR(20) DEFAULT 'user',
            is_active BOOLEAN DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # DoctorDiagnosis tablosunu oluştur
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS doctor_diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            appointment_id INTEGER NOT NULL,
            doctor_id INTEGER NOT NULL,
            age INTEGER NOT NULL,
            gender INTEGER NOT NULL,
            height REAL,
            weight REAL,
            ap_hi INTEGER NOT NULL,
            ap_lo INTEGER NOT NULL,
            cholesterol INTEGER NOT NULL,
            gluc INTEGER NOT NULL,
            smoke INTEGER NOT NULL,
            alco INTEGER NOT NULL,
            active INTEGER NOT NULL,
            risk_score REAL NOT NULL,
            priority_score REAL NOT NULL,
            prediction_label INTEGER NOT NULL,
            prediction_probability REAL NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            doctor_target INTEGER NOT NULL,
            doctor_notes TEXT,
            diagnosis_status VARCHAR(20) DEFAULT 'completed',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (appointment_id) REFERENCES appointments (id),
            FOREIGN KEY (doctor_id) REFERENCES users (id)
        )
        ''')
        
        # Test verisi ekle
        cursor.execute('''
        INSERT OR IGNORE INTO users (id, username, email, full_name, hashed_password, role) 
        VALUES (1, 'testdoctor', 'doctor@example.com', 'Dr. Test Doctor', 'hashed_password', 'doctor')
        ''')
        
        cursor.execute('''
        INSERT OR IGNORE INTO users (id, username, email, full_name, hashed_password, role) 
        VALUES (13, 'testpatient123', 'testpatient@example.com', 'Test Patient', 'hashed_password', 'user')
        ''')
        
        # Test appointment verisi ekle
        cursor.execute('''
        INSERT OR IGNORE INTO appointments (
            id, patient_id, prediction_record_id, age, gender, height, weight,
            ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active,
            risk_score, priority_score, prediction_label, prediction_probability,
            model_name, status, priority, created_at
        ) VALUES (
            1, 13, 1, 45, 1, 175.0, 80.0, 120, 80, 1, 1, 0, 0, 1,
            0.85, 0.9, 1, 0.85, 'TestModel', 'completed', 'high', CURRENT_TIMESTAMP
        )
        ''')
        
        # Test doctor diagnosis verisi ekle
        cursor.execute('''
        INSERT OR IGNORE INTO doctor_diagnoses (
            appointment_id, doctor_id, age, gender, height, weight,
            ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active,
            risk_score, priority_score, prediction_label, prediction_probability,
            model_name, doctor_target, doctor_notes, diagnosis_status
        ) VALUES (
            1, 1, 45, 1, 175.0, 80.0, 120, 80, 1, 1, 0, 0, 1,
            0.85, 0.9, 1, 0.85, 'TestModel', 1, 'Hasta yüksek risk grubunda. Düzenli takip gerekli.', 'completed'
        )
        ''')
        
        conn.commit()
        conn.close()
        
        print("Tablolar başarıyla oluşturuldu!")
        print("Test verileri eklendi!")
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    create_appointments_table()
