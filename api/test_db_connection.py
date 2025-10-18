"""
Veritabanı bağlantı testi
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.core.database import engine, Base, SessionLocal, User
from api.services.auth_service import get_password_hash
from sqlalchemy.exc import SQLAlchemyError

def test_database_connection():
    """Veritabanı bağlantısını test et."""
    try:
        # Tabloları oluştur
        Base.metadata.create_all(bind=engine)
        print("✅ Veritabanı tabloları oluşturuldu")
        
        # Bağlantıyı test et
        db = SessionLocal()
        from sqlalchemy import text
        result = db.execute(text("SELECT 1"))
        print("✅ PostgreSQL bağlantısı başarılı")
        
        # Test kullanıcısı oluştur
        test_user = db.query(User).filter(User.username == "admin").first()
        if not test_user:
            hashed_password = get_password_hash("admin")
            test_user = User(
                username="admin",
                email="admin@example.com",
                password_hash=hashed_password,
                full_name="Admin User",
                role="admin"
            )
            db.add(test_user)
            db.commit()
            print("✅ Test kullanıcısı oluşturuldu (admin/admin)")
        else:
            print("✅ Test kullanıcısı zaten mevcut")
        
        db.close()
        return True
        
    except SQLAlchemyError as e:
        print(f"❌ Veritabanı hatası: {e}")
        return False
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        return False

if __name__ == "__main__":
    print("🔍 PostgreSQL bağlantısı test ediliyor...")
    success = test_database_connection()
    
    if success:
        print("\n🎉 Tüm testler başarılı!")
        print("📋 API'yi başlatmak için: uvicorn main:app --reload")
    else:
        print("\n💥 Test başarısız! Lütfen veritabanı ayarlarını kontrol edin.")
