"""
API Konfigürasyon Modülü
"""

from pydantic_settings import BaseSettings
from pathlib import Path
import os

class Settings(BaseSettings):
    """Uygulama ayarları."""
    
    # API Ayarları
    app_name: str = "Kalp Krizi Risk Tahmin API"
    debug: bool = True
    version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Veritabanı Ayarları
    database_url: str = "postgresql://postgres:12345@localhost:5432/cardiyovask_db"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "cardiyovask_db"
    db_user: str = "postgres"
    db_password: str = "12345"
    
    # JWT Ayarları
    SECRET_KEY: str = "your-secret-key-change-this-in-production-12345"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Ayarları
    cors_origins: str = "*"  # Production'da spesifik domainler belirtilmeli
    cors_allow_credentials: bool = True
    cors_allow_methods: str = "*"
    cors_allow_headers: str = "*"
    
    # Model Ayarları
    models_dir: str = "models"
    data_file: str = "src/data/cardiokaggle.csv"
    
    # GPU Ayarları
    use_gpu: bool = False
    gpu_device_id: int = 0
    
    # Model Parametreleri
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    use_advanced_features: bool = True
    
    # Dosya Yolları
    project_root: Path = Path(__file__).parent.parent.parent
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
