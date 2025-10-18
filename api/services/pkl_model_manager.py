"""
PKL Model Yönetimi Servisi
PKL dosyalarından model bilgilerini okuyup veritabanına kaydetmek için
"""

import pickle
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.core.database import SessionLocal, ModelRecord
from api.core.config import settings

class PKLModelManager:
    """PKL dosyalarından model bilgilerini yöneten servis."""
    
    def __init__(self):
        self.models_dir = project_root / "models"
        # Alternatif PKL klasörleri de kontrol et
        self.alternative_dirs = [
            project_root / "src" / "models",
            project_root / "models",
            project_root / "data" / "models"
        ]
    
    def scan_pkl_files(self) -> List[Dict[str, Any]]:
        """PKL dosyalarını tarayıp bilgilerini döndür."""
        pkl_files = []
        
        # Tüm alternatif klasörleri kontrol et
        for models_dir in self.alternative_dirs:
            if models_dir.exists():
                print(f"🔍 PKL dosyaları aranıyor: {models_dir}")
                for pkl_file in models_dir.glob("*.pkl"):
                    try:
                        model_info = self._extract_model_info(pkl_file)
                        if model_info:
                            pkl_files.append(model_info)
                            print(f"PKL dosyasi bulundu: {pkl_file.name}")
                    except Exception as e:
                        print(f"{pkl_file.name} okunamadi: {e}")
        
        if not pkl_files:
            print(f"Hicbir PKL dosyasi bulunamadi. Kontrol edilen klasorler:")
            for models_dir in self.alternative_dirs:
                print(f"   - {models_dir}")
        
        return pkl_files
    
    def _extract_model_info(self, pkl_file: Path) -> Optional[Dict[str, Any]]:
        """PKL dosyasından model bilgilerini çıkar."""
        try:
            with open(pkl_file, 'rb') as f:
                model_data = pickle.load(f)
            
            # Model bilgilerini çıkar
            model_name = model_data.get('model_name', 'Unknown')
            data_type = model_data.get('data_type', 'unknown')
            timestamp = model_data.get('timestamp', '')
            metadata = model_data.get('metadata', {})
            
            # Dosya boyutu ve tarihi
            file_stats = pkl_file.stat()
            file_size_mb = file_stats.st_size / (1024 * 1024)
            file_date = datetime.fromtimestamp(file_stats.st_mtime)
            
            return {
                'file_path': str(pkl_file),
                'file_name': pkl_file.name,
                'model_name': model_name,
                'data_type': data_type,
                'timestamp': timestamp,
                'file_size_mb': round(file_size_mb, 2),
                'file_date': file_date,
                'metadata': metadata,
                'model_exists': 'model' in model_data
            }
            
        except Exception as e:
            print(f"{pkl_file.name} parse edilemedi: {e}")
            return None
    
    def sync_pkl_to_database(self) -> Dict[str, Any]:
        """PKL dosyalarını veritabanına senkronize et."""
        results = {
            'scanned_files': 0,
            'new_models': 0,
            'updated_models': 0,
            'errors': []
        }
        
        db = SessionLocal()
        try:
            pkl_files = self.scan_pkl_files()
            results['scanned_files'] = len(pkl_files)
            
            for pkl_info in pkl_files:
                try:
                    # Veritabanında bu dosya var mı kontrol et
                    existing_model = db.query(ModelRecord).filter(
                        ModelRecord.model_path == pkl_info['file_path']
                    ).first()
                    
                    if existing_model:
                        # Mevcut modeli güncelle
                        existing_model.created_at = pkl_info['file_date']
                        db.commit()
                        results['updated_models'] += 1
                        print(f"Guncellendi: {pkl_info['file_name']}")
                    else:
                        # Yeni model kaydet
                        model_record = ModelRecord(
                            model_name=f"{pkl_info['model_name']}_{pkl_info['data_type']}",
                            model_type=self._get_model_type(pkl_info['model_name']),
                            accuracy=0.0,  # PKL'den çıkarılamıyor
                            precision=0.0,
                            recall=0.0,
                            f1_score=0.0,
                            roc_auc=None,
                            model_path=pkl_info['file_path'],
                            feature_names=json.dumps([]),
                            training_params=json.dumps(pkl_info['metadata']),
                            created_at=pkl_info['file_date'],
                            is_active=True
                        )
                        
                        db.add(model_record)
                        db.commit()
                        db.refresh(model_record)
                        results['new_models'] += 1
                        print(f"Yeni kayit: {pkl_info['file_name']} (ID: {model_record.id})")
                        
                except Exception as e:
                    error_msg = f"{pkl_info['file_name']}: {str(e)}"
                    results['errors'].append(error_msg)
                    print(f"Hata: {error_msg}")
                    db.rollback()
            
        except Exception as e:
            print(f"Genel hata: {e}")
            results['errors'].append(f"Genel hata: {str(e)}")
        finally:
            db.close()
        
        return results
    
    def _get_model_type(self, model_name: str) -> str:
        """Model tipini belirle."""
        model_name_lower = model_name.lower()
        if any(x in model_name_lower for x in ['random forest', 'gradient boosting', 'xgboost', 'lightgbm', 'catboost']):
            return 'tree_based'
        elif any(x in model_name_lower for x in ['logistic regression', 'svm']):
            return 'linear'
        elif 'ensemble' in model_name_lower:
            return 'ensemble'
        else:
            return 'other'
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Model istatistiklerini döndür."""
        pkl_files = self.scan_pkl_files()
        
        if not pkl_files:
            return {
                'total_files': 0,
                'total_size_mb': 0,
                'model_types': {},
                'date_range': None
            }
        
        # İstatistikleri hesapla
        total_size = sum(f['file_size_mb'] for f in pkl_files)
        model_types = {}
        dates = [f['file_date'] for f in pkl_files]
        
        for file_info in pkl_files:
            model_type = self._get_model_type(file_info['model_name'])
            model_types[model_type] = model_types.get(model_type, 0) + 1
        
        return {
            'total_files': len(pkl_files),
            'total_size_mb': round(total_size, 2),
            'model_types': model_types,
            'date_range': {
                'oldest': min(dates).isoformat() if dates else None,
                'newest': max(dates).isoformat() if dates else None
            }
        }

# Global instance
pkl_model_manager = PKLModelManager()
