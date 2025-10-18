"""
Test Admin Kullanıcısı Oluşturma Scripti
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from api.core.database import SessionLocal, User
from api.services.auth_service import get_password_hash

def create_test_admin():
    """Test admin kullanıcısı oluştur."""
    db = SessionLocal()
    try:
        # Admin kullanıcısı var mı kontrol et
        admin_user = db.query(User).filter(User.username == "admin").first()
        
        if admin_user:
            print("✅ Admin kullanıcısı zaten mevcut")
            return admin_user
        
        # Yeni admin kullanıcısı oluştur
        hashed_password = get_password_hash("admin123")
        
        admin_user = User(
            username="admin",
            email="admin@cardiyovask.com",
            password_hash=hashed_password,
            full_name="Sistem Yöneticisi",
            role="admin",
            is_active=True
        )
        
        db.add(admin_user)
        db.commit()
        db.refresh(admin_user)
        
        print("✅ Admin kullanıcısı oluşturuldu:")
        print(f"   Kullanıcı Adı: admin")
        print(f"   Şifre: admin123")
        print(f"   Email: admin@cardiyovask.com")
        print(f"   Rol: admin")
        
        return admin_user
        
    except Exception as e:
        print(f"❌ Admin kullanıcısı oluşturma hatası: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def create_test_user():
    """Test kullanıcısı oluştur."""
    db = SessionLocal()
    try:
        # Test kullanıcısı var mı kontrol et
        test_user = db.query(User).filter(User.username == "testuser").first()
        
        if test_user:
            print("✅ Test kullanıcısı zaten mevcut")
            return test_user
        
        # Yeni test kullanıcısı oluştur
        hashed_password = get_password_hash("test123")
        
        test_user = User(
            username="testuser",
            email="test@cardiyovask.com",
            password_hash=hashed_password,
            full_name="Test Kullanıcısı",
            role="user",
            is_active=True
        )
        
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        print("✅ Test kullanıcısı oluşturuldu:")
        print(f"   Kullanıcı Adı: testuser")
        print(f"   Şifre: test123")
        print(f"   Email: test@cardiyovask.com")
        print(f"   Rol: user")
        
        return test_user
        
    except Exception as e:
        print(f"❌ Test kullanıcısı oluşturma hatası: {e}")
        db.rollback()
        return None
    finally:
        db.close()

def create_test_doctor():
    """Test doktor kullanıcısı oluştur."""
    db = SessionLocal()
    try:
        # Test doktoru var mı kontrol et
        test_doctor = db.query(User).filter(User.username == "testdoctor").first()
        
        if test_doctor:
            print("✅ Test doktoru zaten mevcut")
            return test_doctor
        
        # Yeni test doktoru oluştur
        hashed_password = get_password_hash("doctor123")
        
        test_doctor = User(
            username="testdoctor",
            email="doctor@cardiyovask.com",
            password_hash=hashed_password,
            full_name="Test Doktoru",
            role="doctor",
            is_active=True
        )
        
        db.add(test_doctor)
        db.commit()
        db.refresh(test_doctor)
        
        print("✅ Test doktoru oluşturuldu:")
        print(f"   Kullanıcı Adı: testdoctor")
        print(f"   Şifre: doctor123")
        print(f"   Email: doctor@cardiyovask.com")
        print(f"   Rol: doctor")
        
        return test_doctor
        
    except Exception as e:
        print(f"❌ Test doktoru oluşturma hatası: {e}")
        db.rollback()
        return None
    finally:
        db.close()

if __name__ == "__main__":
    print("🔐 Test Kullanıcıları Oluşturuluyor...")
    print("=" * 50)
    
    # Admin kullanıcısı oluştur
    create_test_admin()
    print()
    
    # Test kullanıcısı oluştur
    create_test_user()
    print()
    
    # Test doktoru oluştur
    create_test_doctor()
    print()
    
    print("=" * 50)
    print("✅ Tüm test kullanıcıları hazır!")
    print("\n📋 Giriş Bilgileri:")
    print("Admin: admin / admin123")
    print("Doktor: testdoctor / doctor123")
    print("Kullanıcı: testuser / test123")
