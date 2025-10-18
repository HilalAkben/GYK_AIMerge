"""
Veritabanı Tablolarını Yeniden Oluştur
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.core.database import Base, engine, SessionLocal

def reset_database():
    """Veritabanı tablolarını yeniden oluştur."""
    try:
        print("🗑️ Mevcut tablolar siliniyor...")
        Base.metadata.drop_all(bind=engine)
        
        print("🏗️ Yeni tablolar oluşturuluyor...")
        Base.metadata.create_all(bind=engine)
        
        print("✅ Veritabanı başarıyla yeniden oluşturuldu!")
        
        # Test bağlantısı
        db = SessionLocal()
        from sqlalchemy import text
        result = db.execute(text("SELECT 1"))
        db.close()
        print("✅ Veritabanı bağlantısı test edildi")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    reset_database()
