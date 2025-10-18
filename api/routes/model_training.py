"""
Model Eğitimi Router
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Body, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.services.model_training_service import model_training_service
from api.services.auth_service import require_admin, require_doctor_or_admin
from api.core.config import settings

router = APIRouter()

class ModelTrainingRequest(BaseModel):
    """Model eğitimi isteği."""
    model_name: str
    data_path: Optional[str] = None
    use_advanced_features: bool = True
    use_gpu: bool = settings.use_gpu
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "Random Forest",
                "data_path": "src/data/cardiokaggle.csv",
                "use_advanced_features": True,
                "use_gpu": False
            }
        }

class ModelTrainingResponse(BaseModel):
    """Model eğitimi yanıtı."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Model eğitimi başarıyla tamamlandı",
                "data": {
                    "training_summary": {
                        "model_name": "Random Forest",
                        "model_type": "tree_based",
                        "data_shape": {
                            "train": [56042, 12],
                            "test": [14011, 12],
                            "features": 12
                        }
                    },
                    "evaluation_results": {
                        "accuracy": 0.8156,
                        "precision": 0.8123,
                        "recall": 0.8345,
                        "f1_score": 0.8234,
                        "roc_auc": 0.8456
                    },
                    "model_path": "models/random_forest_20241201_143022.pkl",
                    "model_record_id": 1
                },
                "execution_time": 23.45
            }
        }

class ModelListResponse(BaseModel):
    """Model listesi yanıtı."""
    success: bool
    message: str
    models: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Modeller başarıyla listelendi",
                "models": [
                    {
                        "id": 1,
                        "model_name": "Random Forest",
                        "model_type": "tree_based",
                        "accuracy": 0.8156,
                        "f1_score": 0.8234,
                        "created_at": "2024-12-01T14:30:22",
                        "feature_count": 12
                    }
                ]
            }
        }

@router.post("/train", response_model=ModelTrainingResponse)
async def train_model(request: ModelTrainingRequest, current_user = Depends(require_doctor_or_admin)):
    """
    Belirtilen modeli eğit ve detaylı analiz sağla.
    
    Bu endpoint:
    - Belirtilen modeli eğitir
    - Detaylı performans analizi yapar
    - Feature importance analizi sağlar
    - Modeli kaydeder ve veritabanına kayıt eder
    
    **Desteklenen Modeller:**
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Logistic Regression
    - SVM
    - CatBoost
    - AdaBoost
    - Ensemble Methods
    
    **Özellikler:**
    - Gelişmiş feature engineering
    - GPU desteği
    - Detaylı analiz raporu
    - Model kaydetme
    """
    try:
        import time
        start_time = time.time()

        # Veri yolunu belirle
        if request.data_path:
            # Relative path ise project_root'a göre ayarla
            if not Path(request.data_path).is_absolute():
                data_path = str(settings.project_root / request.data_path)
            else:
                data_path = request.data_path
        else:
            data_path = str(settings.project_root / settings.data_file)
        
        # Dosya varlığını kontrol et
        if not Path(data_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Veri dosyası bulunamadı: {data_path}"
            )

        # Model eğitimini çalıştır
        result = await model_training_service.train_selected_model(
            model_name=request.model_name,
            data_path=data_path,
            use_advanced_features=request.use_advanced_features,
            use_gpu=request.use_gpu
        )
        
        execution_time = time.time() - start_time
        
        return ModelTrainingResponse(
            success=True,
            message=f"{request.model_name} modeli başarıyla eğitildi",
            data=result,
            execution_time=execution_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model eğitimi hatası: {str(e)}"
        )

@router.post("/train/auto", response_model=ModelTrainingResponse)
async def auto_train_best_model(current_user = Depends(require_admin)):
    """
    Ek parametre istemeden, kalibrasyon temelli metriklere göre en iyi modeli seçip eğitir ve kaydeder.
    - /api/v1/models/best mantığını kullanır (Brier, LogLoss, ECE)
    - Varsayılan veri dosyası yolunu kullanır
    """
    try:
        import time
        start_time = time.time()

        # Varsayılan veri yolu
        data_path = str(settings.project_root / settings.data_file)
        if not Path(data_path).exists():
            raise HTTPException(status_code=404, detail=f"Veri dosyası bulunamadı: {data_path}")

        # En iyi modeli seç (with/without outliers ayrı döner)
        from api.services.model_comparison_service import model_comparison_service
        selection = await model_comparison_service.select_best_model_by_calibration(data_path, n_bins=10)

        # Öncelik: without_outliers, yoksa with_outliers
        best = selection.get('without_outliers', {}).get('best_model') or selection.get('with_outliers', {}).get('best_model')
        if not best:
            raise HTTPException(status_code=400, detail="En iyi model belirlenemedi.")

        # Eğit
        result = await model_training_service.train_selected_model(
            model_name=best,
            data_path=data_path,
            use_advanced_features=True,
            use_gpu=True
        )

        return ModelTrainingResponse(
            success=True,
            message=f"Otomatik en iyi model ({best}) başarıyla eğitildi",
            data=result,
            execution_time=time.time() - start_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Otomatik eğitim hatası: {str(e)}")

@router.get("/best-model", response_model=ModelTrainingResponse)
async def get_best_model():
    """
    Veritabanından en iyi performanslı modeli getir (yeniden eğitim yapmaz).
    F1 skoruna göre en iyi modeli döndürür.
    """
    try:
        from api.services.prediction_service import prediction_service
        
        # Veritabanından en iyi modeli getir
        best_model = await prediction_service.get_best_model_from_database()
        
        if not best_model:
            raise HTTPException(
                status_code=404, 
                detail="Veritabanında aktif model bulunamadı"
            )
        
        # Model bilgilerini formatla
        result = {
            'model_info': {
                'model_name': best_model['model_name'],
                'model_type': best_model['model_type'],
                'model_id': best_model['id'],
                'created_at': best_model['created_at']
            },
            'evaluation_results': {
                'accuracy': best_model['accuracy'],
                'precision': best_model['precision'],
                'recall': best_model['recall'],
                'f1_score': best_model['f1_score'],
                'roc_auc': best_model['roc_auc']
            },
            'training_summary': {
                'model_name': best_model['model_name'],
                'feature_count': len(best_model['feature_names']),
                'training_params': best_model['training_params']
            },
            'feature_names': best_model['feature_names'],
            'model_path': best_model['model_path']
        }
        
        return ModelTrainingResponse(
            success=True,
            message=f"En iyi model getirildi: {best_model['model_name']} (F1: {best_model['f1_score']:.4f})",
            data=result,
            execution_time=0.0  # Veritabanından okuma, eğitim yok
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"En iyi model getirme hatası: {str(e)}"
        )

@router.delete("/models/{model_id}")
async def delete_model(model_id: int, current_user = Depends(require_admin)):
    """
    Belirtilen modeli hem veritabanından hem PKL dosyasından sil.
    
    Bu endpoint:
    - Modeli veritabanından siler
    - PKL dosyasını fiziksel olarak siler
    - İlişkili prediction kayıtlarını kontrol eder
    """
    try:
        from api.services.prediction_service import prediction_service
        from api.core.database import SessionLocal, ModelRecord, PredictionRecord
        import os
        
        db = SessionLocal()
        try:
            # Modeli veritabanından bul
            model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
            
            if not model:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Model ID {model_id} bulunamadı"
                )
            
            # İlişkili prediction kayıtlarını kontrol et
            prediction_count = db.query(PredictionRecord).filter(
                PredictionRecord.model_id == model_id
            ).count()
            
            if prediction_count > 0:
                # Prediction kayıtlarını da sil
                db.query(PredictionRecord).filter(
                    PredictionRecord.model_id == model_id
                ).delete()
                print(f"🗑️ {prediction_count} prediction kaydı silindi")
            
            # PKL dosyasını sil
            model_path = model.model_path
            if model_path and os.path.exists(model_path):
                try:
                    os.remove(model_path)
                    print(f"🗑️ PKL dosyası silindi: {model_path}")
                except Exception as e:
                    print(f"⚠️ PKL dosyası silinemedi: {e}")
            
            # Modeli veritabanından sil
            db.delete(model)
            db.commit()
            
            return {
                "success": True,
                "message": f"Model '{model.model_name}' başarıyla silindi",
                "data": {
                    "deleted_model_id": model_id,
                    "deleted_model_name": model.model_name,
                    "deleted_pkl_file": model_path,
                    "deleted_predictions": prediction_count
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Model silme hatası: {str(e)}"
            )
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model silme hatası: {str(e)}"
        )

@router.get("/models", response_model=ModelListResponse)
async def get_trained_models():
    """
    Eğitilmiş modelleri listele.
    
    Bu endpoint, veritabanında kayıtlı tüm aktif modelleri listeler.
    """
    try:
        from api.core.database import SessionLocal, ModelRecord
        
        db = SessionLocal()
        try:
            models = db.query(ModelRecord).filter(ModelRecord.is_active == True).all()
            
            model_list = []
            for model in models:
                model_list.append({
                    "id": model.id,
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "accuracy": model.accuracy,
                    "precision": model.precision,
                    "recall": model.recall,
                    "f1_score": model.f1_score,
                    "roc_auc": model.roc_auc,
                    "created_at": model.created_at.isoformat(),
                    "feature_count": len(model.feature_names) if model.feature_names else 0,
                    "model_path": model.model_path
                })
            
            return ModelListResponse(
                success=True,
                message=f"{len(model_list)} model bulundu",
                models=model_list
            )
            
        finally:
            db.close()
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model listesi hatası: {str(e)}"
        )

@router.get("/models/{model_id}")
async def get_model_details(model_id: int):
    """
    Belirli bir modelin detaylarını getir.
    
    Bu endpoint, belirtilen model ID'sine sahip modelin detaylı bilgilerini döndürür.
    """
    try:
        from api.core.database import SessionLocal, ModelRecord
        
        db = SessionLocal()
        try:
            model = db.query(ModelRecord).filter(
                ModelRecord.id == model_id,
                ModelRecord.is_active == True
            ).first()
            
            if not model:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model ID {model_id} bulunamadı"
                )
            
            return {
                "success": True,
                "message": "Model detayları başarıyla getirildi",
                "model": {
                    "id": model.id,
                    "model_name": model.model_name,
                    "model_type": model.model_type,
                    "accuracy": model.accuracy,
                    "precision": model.precision,
                    "recall": model.recall,
                    "f1_score": model.f1_score,
                    "roc_auc": model.roc_auc,
                    "model_path": model.model_path,
                    "feature_names": model.feature_names,
                    "training_params": model.training_params,
                    "created_at": model.created_at.isoformat(),
                    "is_active": model.is_active
                }
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model detay hatası: {str(e)}"
        )

@router.delete("/models/{model_id}")
async def delete_model(model_id: int):
    """
    Belirli bir modeli sil (soft delete).
    
    Bu endpoint, belirtilen model ID'sine sahip modeli pasif hale getirir.
    """
    try:
        from api.core.database import SessionLocal, ModelRecord
        
        db = SessionLocal()
        try:
            model = db.query(ModelRecord).filter(ModelRecord.id == model_id).first()
            
            if not model:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model ID {model_id} bulunamadı"
                )
            
            # Soft delete - is_active'i False yap
            model.is_active = False
            db.commit()
            
            return {
                "success": True,
                "message": f"Model {model.model_name} başarıyla silindi"
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model silme hatası: {str(e)}"
        )

@router.get("/supported-models")
async def get_supported_models():
    """
    Desteklenen modelleri listele.
    
    Bu endpoint, eğitilebilecek tüm model türlerini ve özelliklerini döndürür.
    """
    supported_models = {
        "tree_based": [
            {
                "name": "Random Forest",
                "description": "Çoklu karar ağacı ensemble metodu",
                "strengths": ["Yüksek doğruluk", "Feature importance", "Outlier'a dayanıklı"],
                "weaknesses": ["Overfitting riski", "Yavaş tahmin"],
                "best_for": "Genel amaçlı sınıflandırma"
            },
            {
                "name": "Gradient Boosting",
                "description": "Gradient boosting ensemble metodu",
                "strengths": ["Yüksek performans", "Feature importance", "Non-linear ilişkiler"],
                "weaknesses": ["Overfitting riski", "Yavaş eğitim"],
                "best_for": "Yüksek doğruluk gereken durumlar"
            },
            {
                "name": "XGBoost",
                "description": "Extreme Gradient Boosting",
                "strengths": ["En yüksek performans", "GPU desteği", "Hızlı eğitim"],
                "weaknesses": ["Karmaşık parametreler", "Overfitting riski"],
                "best_for": "Competition ve production ortamları"
            },
            {
                "name": "LightGBM",
                "description": "Light Gradient Boosting Machine",
                "strengths": ["Hızlı eğitim", "Düşük bellek kullanımı", "GPU desteği"],
                "weaknesses": ["Küçük veri setlerinde overfitting"],
                "best_for": "Büyük veri setleri"
            }
        ],
        "linear": [
            {
                "name": "Logistic Regression",
                "description": "Doğrusal sınıflandırma modeli",
                "strengths": ["Hızlı", "Yorumlanabilir", "Overfitting riski düşük"],
                "weaknesses": ["Doğrusal ilişkiler varsayımı", "Düşük performans"],
                "best_for": "Baseline model ve yorumlanabilirlik"
            },
            {
                "name": "SVM",
                "description": "Support Vector Machine",
                "strengths": ["Küçük veri setlerinde iyi", "Non-linear kernel desteği"],
                "weaknesses": ["Büyük veri setlerinde yavaş", "Yorumlanabilir değil"],
                "best_for": "Küçük-orta boyutlu veri setleri"
            }
        ]
    }
    
    return {
        "success": True,
        "message": "Desteklenen modeller listelendi",
        "supported_models": supported_models,
        "total_count": sum(len(models) for models in supported_models.values())
    }

@router.get("/training/status")
async def get_training_status():
    """
    Model eğitimi durumunu kontrol et.
    
    Bu endpoint, model eğitimi servisinin durumunu döndürür.
    """
    return {
        "status": "ready",
        "message": "Model eğitimi servisi hazır",
        "features": {
            "advanced_feature_engineering": True,
            "gpu_support": True,
            "model_persistence": True,
            "database_integration": True
        },
        "data_file": str(settings.project_root / settings.data_file)
    }
