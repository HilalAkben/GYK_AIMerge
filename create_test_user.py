"""
Test Kullanıcısı Oluştur
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from api.core.database import SessionLocal, User
from api.services.auth_service import get_password_hash

def create_test_user():
    """Test kullanıcısı oluştur."""
    db = SessionLocal()
    try:
        # Admin kullanıcısını kontrol et
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            hashed_password = get_password_hash("123"[:72])
            admin_user = User(
                username="admin",
                email="admin@example.com",
                password_hash=hashed_password,
                full_name="Admin User",
                role="admin"
            )
            db.add(admin_user)
            db.commit()
            print("Admin kullanici olusturuldu (admin/123)")
        else:
            print("Admin kullanici zaten mevcut")
        
        # Doctor kullanıcısı oluştur
        doctor_user = db.query(User).filter(User.username == "doctor").first()
        if not doctor_user:
            hashed_password = get_password_hash("123"[:72])
            doctor_user = User(
                username="doctor",
                email="doctor@example.com",
                password_hash=hashed_password,
                full_name="Dr. Ahmet Yılmaz",
                role="doctor"
            )
            db.add(doctor_user)
            db.commit()
            print("Doctor kullanici olusturuldu (doctor/123)")
        else:
            print("Doctor kullanici zaten mevcut")
            
        # Test kullanıcısı oluştur
        test_user = db.query(User).filter(User.username == "kullanıcı").first()
        if not test_user:
            hashed_password = get_password_hash("123"[:72])
            test_user = User(
                username="kullanıcı",
                email="kullanici@example.com",
                password_hash=hashed_password,
                full_name="Test Kullanıcısı",
                role="user"
            )
            db.add(test_user)
            db.commit()
            print("Test kullanici olusturuldu (kullanici/123)")
        else:
            print("Test kullanici zaten mevcut")
            
        # Test doktoru oluştur
        doctor_user = db.query(User).filter(User.username == "doktor").first()
        if not doctor_user:
            hashed_password = get_password_hash("123"[:72])
            doctor_user = User(
                username="doktor",
                email="doktor@example.com",
                password_hash=hashed_password,
                full_name="Dr. Test Doktoru",
                role="doctor"
            )
            db.add(doctor_user)
            db.commit()
            print("Test doktoru olusturuldu (doktor/123)")
        else:
            print("Test doktoru zaten mevcut")
            
        # Test hastası oluştur
        patient_user = db.query(User).filter(User.username == "hasta").first()
        if not patient_user:
            hashed_password = get_password_hash("123"[:72])
            patient_user = User(
                username="hasta",
                email="hasta@example.com",
                password_hash=hashed_password,
                full_name="Test Hastası",
                role="user"
            )
            db.add(patient_user)
            db.commit()
            print("Test hastasi olusturuldu (hasta/123)")
        else:
            print("Test hastasi zaten mevcut")
            
    except Exception as e:
        print(f"Kullanici olusturma hatasi: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_test_user()
