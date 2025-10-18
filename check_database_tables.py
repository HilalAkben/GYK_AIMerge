#!/usr/bin/env python3
"""
Veritabanındaki tabloları kontrol etmek için script
"""

import sqlite3

def check_database_tables():
    try:
        # Veritabanına bağlan
        conn = sqlite3.connect('cardio_models.db')
        cursor = conn.cursor()
        
        # Tabloları listele
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("Mevcut Tablolar:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Her tablo için satır sayısını kontrol et
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  {table_name}: {count} satır")
        
        conn.close()
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    check_database_tables()
