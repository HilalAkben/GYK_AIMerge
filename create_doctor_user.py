"""
Test doktor kullanıcısı oluşturma scripti
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from api.core.database import SessionLocal, User
from api.services.auth_service import get_password_hash

def create_doctor_user():
    """Test doktor kullanıcısı oluştur."""
    db = SessionLocal()
    try:
        # Test doktoru oluştur
        doctor_user = db.query(User).filter(User.username == "doktor").first()
        if not doctor_user:
            hashed_password = get_password_hash("123")
            doctor_user = User(
                username="doktor",
                email="doktor@example.com",
                password_hash=hashed_password,
                full_name="Dr. Test Doktoru",
                role="doctor"
            )
            db.add(doctor_user)
            db.commit()
            print("✅ Test doktoru oluşturuldu (doktor/123)")
        else:
            print("✅ Test doktoru zaten mevcut")
            
        # Test hastası oluştur
        patient_user = db.query(User).filter(User.username == "hasta").first()
        if not patient_user:
            hashed_password = get_password_hash("123")
            patient_user = User(
                username="hasta",
                email="hasta@example.com",
                password_hash=hashed_password,
                full_name="Test Hastası",
                role="user"
            )
            db.add(patient_user)
            db.commit()
            print("✅ Test hastası oluşturuldu (hasta/123)")
        else:
            print("✅ Test hastası zaten mevcut")
            
        # Test admin'i güncelle
        admin_user = db.query(User).filter(User.username == "adminn").first()
        if admin_user:
            admin_user.role = "admin"
            db.commit()
            print("✅ Admin kullanıcısı güncellendi")
        else:
            print("❌ Admin kullanıcısı bulunamadı")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_doctor_user()
