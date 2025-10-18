#!/usr/bin/env python3
"""
Appointments tablosundaki verileri kontrol etmek için script
"""

import sqlite3
import pandas as pd

def check_appointments_data():
    try:
        # Veritabanına bağlan
        conn = sqlite3.connect('cardio_models.db')
        
        # Appointments tablosundaki verileri kontrol et
        query = "SELECT * FROM appointments LIMIT 10"
        appointments_df = pd.read_sql_query(query, conn)
        
        print("Appointments Tablosu Verileri:")
        print(f"Toplam satır sayısı: {len(appointments_df)}")
        
        if len(appointments_df) > 0:
            print("\nİlk 5 randevu:")
            print(appointments_df.head())
            
            # Patient ID'leri listele
            patient_ids = appointments_df['patient_id'].unique()
            print(f"\nMevcut Patient ID'ler: {patient_ids}")
            
            # Null patient_id'leri kontrol et
            null_patient_ids = appointments_df[appointments_df['patient_id'].isnull()]
            print(f"Null patient_id sayısı: {len(null_patient_ids)}")
            
        else:
            print("Appointments tablosunda veri bulunmuyor")
        
        # Users tablosunu da kontrol et
        users_query = "SELECT id, username, full_name, role FROM users LIMIT 10"
        users_df = pd.read_sql_query(users_query, conn)
        
        print(f"\nUsers Tablosu:")
        print(f"Toplam kullanıcı sayısı: {len(users_df)}")
        if len(users_df) > 0:
            print(users_df.head())
        
        conn.close()
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    check_appointments_data()
