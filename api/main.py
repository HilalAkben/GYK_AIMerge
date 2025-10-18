"""
Kalp Krizi Risk Tahmin API - FastAPI Ana Uygulama
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api.routes import model_comparison, model_training, prediction, auth, appointment_queue, doctor_panel, users
from api.core.config import settings
from api.core.database import init_database
from api.services.scheduler_service import scheduler_service
from api.services.appointment_service import appointment_service

# FastAPI uygulaması oluştur
app = FastAPI(
    title="Kalp Krizi Risk Tahmin API",
    description="""
    ## Kalp Krizi Risk Tahmin Sistemi
    
    Bu API, kardiyovasküler hastalık riskini tahmin etmek için geliştirilmiş makine öğrenmesi modellerini kullanır.
    
    ### Özellikler:
    - **Model Karşılaştırması**: Farklı ML algoritmalarının performansını karşılaştırır
    - **Model Eğitimi**: Seçilen modeli eğitir ve detaylı analiz sağlar
    - **Risk Tahmini**: Hastanın kardiyovasküler risk skorunu hesaplar
    
    ### Desteklenen Modeller:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Logistic Regression
    - SVM
    - Ensemble Methods
    """,
    version="1.0.0",
    contact={
        "name": "Kalp Krizi Risk Tahmin API",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# CORS middleware ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"],
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods.split(",") if settings.cors_allow_methods != "*" else ["*"],
    allow_headers=settings.cors_allow_headers.split(",") if settings.cors_allow_headers != "*" else ["*"],
)

# Router'ları ekle
app.include_router(
    model_comparison.router,
    prefix="/api/v1/models",
    tags=["Model Karşılaştırması"]
)

app.include_router(
    model_training.router,
    prefix="/api/v1/training",
    tags=["Model Eğitimi"]
)

app.include_router(
    prediction.router,
    prefix="/api/v1/prediction",
    tags=["Risk Tahmini"]
)

app.include_router(
    auth.router,
    prefix="/api/v1/auth",
    tags=["Kimlik Doğrulama"]
)

app.include_router(
    appointment_queue.router,
    prefix="/api/v1/admin",
    tags=["Randevu Sırası"]
)

app.include_router(
    doctor_panel.router,
    prefix="/api/v1/doctor",
    tags=["Doktor Paneli"]
)

app.include_router(
    users.router,
    prefix="/api/v1/users",
    tags=["Kullanıcı Profil"]
)

@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında çalışacak fonksiyonlar."""
    print("Kalp Krizi Risk Tahmin API baslatiliyor...")
    
    # Veritabanını başlat
    await init_database()
    
    # Model klasörünü oluştur
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Scheduler'ı başlat
    scheduler_service.start()
    
    # Randevu servisini başlat
    await appointment_service.start()
    
    print("API basariyla baslatildi!")
    print("Gunluk otomatik siralama 23:59'da calisacak")
    print("Randevu atama servisi her 10 dakikada bir calisacak")
    print(f"Swagger UI: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"ReDoc: http://{settings.api_host}:{settings.api_port}/redoc")

@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapatıldığında çalışacak fonksiyonlar."""
    print("API kapatiliyor...")
    
    # Scheduler'ı durdur
    scheduler_service.stop()
    
    # Randevu servisini durdur
    await appointment_service.stop()

@app.get("/", tags=["Ana Sayfa"])
async def root():
    """Ana sayfa - API hakkında bilgi."""
    return {
        "message": "Kalp Krizi Risk Tahmin API'ye hoş geldiniz!",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "model_comparison": "/api/v1/models/compare",
            "model_training": "/api/v1/training/train",
            "prediction": "/api/v1/prediction/predict"
        }
    }

@app.get("/health", tags=["Sistem Durumu"])
async def health_check():
    """Sistem sağlık kontrolü."""
    return {
        "status": "healthy",
        "message": "API çalışıyor",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info"
    )
