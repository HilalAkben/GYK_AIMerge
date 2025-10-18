"""
Model Karşılaştırma Servisi
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.preprocessor import DataPreprocessor
from src.analysis.data_analysis import ComparativeAnalyzer
from api.core.database import ModelRecord, SessionLocal
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

class ModelComparisonService:
    """Model karşılaştırma servisi."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.comparative_analyzer = ComparativeAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def run_model_comparison(self, data_path: str) -> Dict[str, Any]:
        """
        Tüm modelleri karşılaştır ve sonuçları döndür.
        
        Args:
            data_path: Veri dosyası yolu
            
        Returns:
            Model karşılaştırma sonuçları
        """
        try:
            # Asenkron olarak çalıştır
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._run_comparison_sync, 
                data_path
            )
            return result
        except Exception as e:
            raise Exception(f"Model karşılaştırma hatası: {str(e)}")

    async def select_best_model_by_prob_metrics(
        self,
        data_path: str,
        weight_brier: float = 0.5,
        weight_logloss: float = 0.5,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Brier ve Log Loss'un ağırlıklı ortalamasına göre en iyi modeli seç.

        Dönüş:
            {
              'best_model_name': str,
              'scores': { model_name: { 'brier': float, 'log_loss': float, 'combined_score': float } },
              'context': { 'weight_brier': float, 'weight_logloss': float }
            }
        """
        metrics = await self.evaluate_model_metrics(data_path, n_bins=n_bins)

        # metrics yapısı: {'with_outliers': {...}, 'without_outliers': {...}}
        # Üretimde genellikle outliers temizlenmiş veri tercih edilir; yoksa with_outliers'a düş.
        metrics_block = metrics.get('without_outliers') or metrics.get('with_outliers') or {}

        scores: Dict[str, Dict[str, float]] = {}
        for model_name, m in metrics_block.items():
            brier = m.get('brier_score')
            logloss = m.get('log_loss')
            if brier is None or logloss is None:
                continue
            combined = float(weight_brier) * float(brier) + float(weight_logloss) * float(logloss)
            scores[model_name] = {
                'brier': float(brier),
                'log_loss': float(logloss),
                'combined_score': float(combined),
            }

        if not scores:
            raise Exception('Metrikler boş: Brier ve LogLoss hesaplanamadı')

        best_model_name = min(scores.items(), key=lambda kv: kv[1]['combined_score'])[0]

        return {
            'best_model_name': best_model_name,
            'scores': scores,
            'context': {
                'weight_brier': weight_brier,
                'weight_logloss': weight_logloss,
            },
        }

    async def evaluate_model_metrics(self, data_path: str, n_bins: int = 10) -> Dict[str, Any]:
        """Her model için Brier, LogLoss, kalibrasyon eğrileri ve ECE metriklerini döndür."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._compute_metrics_sync,
                data_path,
                n_bins
            )
            return result
        except Exception as e:
            raise Exception(f"Model metrik değerlendirme hatası: {str(e)}")

    async def select_best_model_by_calibration(self, data_path: str, n_bins: int = 10) -> Dict[str, Any]:
        """Brier, LogLoss ve ECE metriklerine göre en iyi modeli otomatik seç."""
        metrics = await self.evaluate_model_metrics(data_path, n_bins)

        def rank_models(metrics_dict: Dict[str, Dict[str, Any]]):
            # Daha düşük daha iyi: brier, log_loss, ece
            models = list(metrics_dict.keys())
            brier_sorted = sorted(models, key=lambda m: (metrics_dict[m]['brier_score'] if metrics_dict[m]['brier_score'] is not None else float('inf')))
            logloss_sorted = sorted(models, key=lambda m: (metrics_dict[m]['log_loss'] if metrics_dict[m]['log_loss'] is not None else float('inf')))
            ece_sorted = sorted(models, key=lambda m: (metrics_dict[m]['ece'] if metrics_dict[m]['ece'] is not None else float('inf')))

            ranks = {}
            for m in models:
                ranks[m] = {
                    'brier_rank': brier_sorted.index(m) + 1 if m in brier_sorted else None,
                    'logloss_rank': logloss_sorted.index(m) + 1 if m in logloss_sorted else None,
                    'ece_rank': ece_sorted.index(m) + 1 if m in ece_sorted else None,
                }
                # Toplam sıralama (None'ları cezalandır)
                total = 0
                count = 0
                for k in ['brier_rank', 'logloss_rank', 'ece_rank']:
                    if ranks[m][k] is not None:
                        total += ranks[m][k]
                        count += 1
                    else:
                        total += len(models) + 1
                ranks[m]['total_rank'] = total if count > 0 else float('inf')
            best = sorted(models, key=lambda m: ranks[m]['total_rank'])[0] if models else None
            return best, ranks

        best_with, ranks_with = rank_models(metrics['with_outliers']) if metrics.get('with_outliers') else (None, {})
        best_without, ranks_without = rank_models(metrics['without_outliers']) if metrics.get('without_outliers') else (None, {})

        selection = {
            'with_outliers': {
                'best_model': best_with,
                'ranks': ranks_with
            },
            'without_outliers': {
                'best_model': best_without,
                'ranks': ranks_without
            }
        }
        return self._convert_numpy_types(selection)
    
    def _run_comparison_sync(self, data_path: str) -> Dict[str, Any]:
        """Senkron model karşılaştırma."""
        print("Model karsilastirmasi baslatiliyor...")
        
        # 1. Veri Ön İşleme
        print("Veri on isleme...")
        
        # Outlier'lı verilerle işleme
        data_with_outliers = self.preprocessor.complete_preprocessing_pipeline(
            data_path, remove_outliers=False
        )
        if data_with_outliers is None:
            raise Exception("Outlier'lı veri işleme başarısız!")
        
        # Outlier'lar çıkarılarak işleme
        data_without_outliers = self.preprocessor.complete_preprocessing_pipeline(
            data_path, remove_outliers=True
        )
        if data_without_outliers is None:
            raise Exception("Outlier'lar çıkarılarak veri işleme başarısız!")
        
        # 2. Veri Hazırlama
        print("🔧 Veri hazırlama...")
        
        def prepare_ml_data(df):
            """Veriyi makine öğrenmesi için hazırla."""
            df_ml = df.copy()
            
            # Kategorik değişkenleri encode et
            categorical_columns = ['gender', 'smoke', 'alco', 'active']
            for col in categorical_columns:
                if col in df_ml.columns:
                    le = LabelEncoder()
                    df_ml[col] = le.fit_transform(df_ml[col])
            
            # Hedef değişkeni ayır
            y = df_ml['cardio']
            X = df_ml.drop(['id', 'age', 'cardio'], axis=1, errors='ignore')
            
            # Sürekli değişkenleri ölçekle
            continuous_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
            available_continuous = [col for col in continuous_columns if col in X.columns]
            
            if available_continuous:
                scaler = StandardScaler()
                X[available_continuous] = scaler.fit_transform(X[available_continuous])
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return X_train, X_test, y_train, y_test, list(X.columns)
        
        # Veri hazırlama
        X_train_with, X_test_with, y_train_with, y_test_with, feature_names_with = prepare_ml_data(data_with_outliers['data'])
        X_train_without, X_test_without, y_train_without, y_test_without, feature_names_without = prepare_ml_data(data_without_outliers['data'])
        
        # Model eğitimi için veri yapısını hazırla
        data_with_outliers_ml = {
            'X_train': X_train_with,
            'X_test': X_test_with,
            'y_train': y_train_with,
            'y_test': y_test_with,
            'feature_names': feature_names_with
        }
        
        data_without_outliers_ml = {
            'X_train': X_train_without,
            'X_test': X_test_without,
            'y_train': y_train_without,
            'y_test': y_test_without,
            'feature_names': feature_names_without
        }
        
        # 3. Karşılaştırmalı Model Analizi
        print("🤖 Model analizi...")
        outlier_results, no_outlier_results = self.comparative_analyzer.run_comparative_analysis(
            data_with_outliers_ml, data_without_outliers_ml
        )
        
        # 4. Olasılık Temelli Metrikler
        print("📈 Olasılık metrikleri hesaplanıyor...")
        
        def compute_probability_metrics(results_dict, y_test, tag):
            """Olasılık temelli metrikleri hesapla."""
            metrics = {}
            
            for model_name, splits in results_dict.items():
                y_proba = splits['test'].get('y_proba', None)
                if y_proba is None:
                    continue
                
                try:
                    brier = brier_score_loss(y_test, y_proba)
                    ll = log_loss(y_test, y_proba)
                    metrics[model_name] = {
                        'brier_score': float(brier),
                        'log_loss': float(ll)
                    }
                except Exception as e:
                    print(f"{model_name}: Olasılık metrikleri hesaplanamadı -> {e}")
                    metrics[model_name] = {
                        'brier_score': None,
                        'log_loss': None
                    }
            
            return metrics
        
        # Olasılık metriklerini hesapla
        prob_metrics_with = compute_probability_metrics(outlier_results['results'], y_test_with, 'with_outliers')
        prob_metrics_without = compute_probability_metrics(no_outlier_results['results'], y_test_without, 'without_outliers')
        
        # 5. Sonuçları Hazırla
        print("Sonuclar hazirlaniyor...")
        
        # Model performanslarını sırala (F1-Score'a göre)
        def sort_models_by_performance(results_dict):
            """Modelleri performansa göre sırala."""
            model_scores = []
            for model_name, splits in results_dict.items():
                test_metrics = splits['test']
                f1_score = test_metrics.get('f1', 0)
                model_scores.append({
                    'model_name': model_name,
                    'f1_score': f1_score,
                    'accuracy': test_metrics.get('accuracy', 0),
                    'precision': test_metrics.get('precision', 0),
                    'recall': test_metrics.get('recall', 0),
                    'roc_auc': test_metrics.get('roc_auc', 0)
                })
            
            # F1-Score'a göre sırala (yüksekten düşüğe)
            model_scores.sort(key=lambda x: x['f1_score'], reverse=True)
            return model_scores
        
        # Outlier'lı ve outlier'sız modelleri sırala
        sorted_models_with = sort_models_by_performance(outlier_results['results'])
        sorted_models_without = sort_models_by_performance(no_outlier_results['results'])
        
        # Risk skorları hesapla (F1-Score'a göre normalize edilmiş)
        def calculate_risk_scores(sorted_models):
            """Risk skorlarını hesapla."""
            if not sorted_models:
                return []
            
            max_f1 = sorted_models[0]['f1_score']
            min_f1 = sorted_models[-1]['f1_score']
            f1_range = max_f1 - min_f1 if max_f1 != min_f1 else 1
            
            risk_scores = []
            for i, model in enumerate(sorted_models):
                # F1-Score'a göre risk skoru (yüksek F1 = düşük risk)
                normalized_f1 = (model['f1_score'] - min_f1) / f1_range
                risk_score = 1 - normalized_f1  # Ters çevir (yüksek performans = düşük risk)
                
                risk_scores.append({
                    'rank': i + 1,
                    'model_name': model['model_name'],
                    'f1_score': model['f1_score'],
                    'accuracy': model['accuracy'],
                    'precision': model['precision'],
                    'recall': model['recall'],
                    'roc_auc': model['roc_auc'],
                    'risk_score': risk_score,
                    'risk_level': self._get_risk_level(risk_score)
                })
            
            return risk_scores
        
        risk_scores_with = calculate_risk_scores(sorted_models_with)
        risk_scores_without = calculate_risk_scores(sorted_models_without)
        
        # Olasılık metriklerine (Brier, LogLoss) göre en iyi modeli seç
        def select_best_by_prob_metrics(prob_metrics: Dict[str, Dict[str, float]]):
            if not prob_metrics:
                return None, None, None
            # Sadece değeri olanları filtrele
            candidates = [
                (name, m.get('brier_score'), m.get('log_loss'))
                for name, m in prob_metrics.items()
                if m.get('brier_score') is not None or m.get('log_loss') is not None
            ]
            if not candidates:
                return None, None, None
            # Önce Brier (küçük daha iyi), eşitlikte LogLoss (küçük daha iyi)
            best = sorted(
                candidates,
                key=lambda x: (
                    float('inf') if x[1] is None else x[1],
                    float('inf') if x[2] is None else x[2]
                )
            )[0]
            return best[0], best[1], best[2]

        best_with_name, best_with_brier, best_with_logloss = select_best_by_prob_metrics(prob_metrics_with)
        best_without_name, best_without_brier, best_without_logloss = select_best_by_prob_metrics(prob_metrics_without)

        # Final sonuçlar
        result = {
            'comparison_summary': {
                'total_models': int(len(outlier_results['results'])),
                'data_with_outliers': {
                    'train_shape': data_with_outliers_ml['X_train'].shape,
                    'test_shape': data_with_outliers_ml['X_test'].shape,
                    # F1'a göre en iyi (geriye dönük uyumluluk için ayrı alan)
                    'best_model_f1': outlier_results['best_model_name'],
                    'best_f1_score': outlier_results['best_score'],
                    # Olasılık metriklerine göre en iyi
                    'best_model': best_with_name,
                    'best_brier_score': best_with_brier,
                    'best_log_loss': best_with_logloss
                },
                'data_without_outliers': {
                    'train_shape': data_without_outliers_ml['X_train'].shape,
                    'test_shape': data_without_outliers_ml['X_test'].shape,
                    'best_model_f1': no_outlier_results['best_model_name'],
                    'best_f1_score': no_outlier_results['best_score'],
                    'best_model': best_without_name,
                    'best_brier_score': best_without_brier,
                    'best_log_loss': best_without_logloss
                }
            },
            'model_rankings': {
                'with_outliers': risk_scores_with,
                'without_outliers': risk_scores_without
            },
            'probability_metrics': {
                'with_outliers': prob_metrics_with,
                'without_outliers': prob_metrics_without
            },
            'feature_importance': {
                'with_outliers': self._extract_feature_importance(outlier_results['importance_dfs']),
                'without_outliers': self._extract_feature_importance(no_outlier_results['importance_dfs'])
            },
            'outlier_analysis': {
                'with_outliers': data_with_outliers['outlier_summary'],
                'without_outliers': data_without_outliers['outlier_summary']
            }
        }
        
        print("Model karsilastirmasi tamamlandi!")
        
        # 5. Modelleri veritabanına kaydet
        print("Modeller veritabanina kaydediliyor...")
        try:
            # Outlier'lı modelleri kaydet
            if 'results' in outlier_results and outlier_results['results']:
                outlier_models = {}
                for model_name, splits in outlier_results['results'].items():
                    outlier_models[model_name] = {
                        'model': splits.get('model'),
                        'accuracy': splits['test']['accuracy'],
                        'precision': splits['test']['precision'],
                        'recall': splits['test']['recall'],
                        'f1_score': splits['test']['f1_score'],
                        'roc_auc': splits['test']['roc_auc'],
                        'feature_names': feature_names_with,
                        'training_params': {'data_type': 'with_outliers'}
                    }
                
                saved_ids_with = self._save_models_to_database(outlier_models, "with_outliers")
                print(f"{len(saved_ids_with)} outlier'li model kaydedildi")
            
            # Outlier'sız modelleri kaydet
            if 'results' in no_outlier_results and no_outlier_results['results']:
                no_outlier_models = {}
                for model_name, splits in no_outlier_results['results'].items():
                    no_outlier_models[model_name] = {
                        'model': splits.get('model'),
                        'accuracy': splits['test']['accuracy'],
                        'precision': splits['test']['precision'],
                        'recall': splits['test']['recall'],
                        'f1_score': splits['test']['f1_score'],
                        'roc_auc': splits['test']['roc_auc'],
                        'feature_names': feature_names_without,
                        'training_params': {'data_type': 'without_outliers'}
                    }
                
                saved_ids_without = self._save_models_to_database(no_outlier_models, "without_outliers")
                print(f"✅ {len(saved_ids_without)} outlier'sız model kaydedildi")
                
        except Exception as e:
            print(f"⚠️ Veritabanı kayıt hatası: {str(e)}")
            # Hata olsa bile devam et
        
        # Numpy/Pandas tiplerini native Python tiplere dönüştür
        result = self._convert_numpy_types(result)
        return result

    def _compute_metrics_sync(self, data_path: str, n_bins: int) -> Dict[str, Any]:
        """Senkron olarak metrikleri hesapla (Brier, LogLoss, kalibrasyon eğrisi, ECE)."""
        # Veri ön işleme ve hazırla (run_comparison ile benzer)
        print("🔄 Metrikler hesaplanıyor...")
        data_with_outliers = self.preprocessor.complete_preprocessing_pipeline(
            data_path, remove_outliers=False
        )
        if data_with_outliers is None:
            raise Exception("Outlier'lı veri işleme başarısız!")
        data_without_outliers = self.preprocessor.complete_preprocessing_pipeline(
            data_path, remove_outliers=True
        )
        if data_without_outliers is None:
            raise Exception("Outlier'lar çıkarılarak veri işleme başarısız!")

        def prepare_ml_data(df):
            df_ml = df.copy()
            categorical_columns = ['gender', 'smoke', 'alco', 'active']
            for col in categorical_columns:
                if col in df_ml.columns:
                    le = LabelEncoder()
                    df_ml[col] = le.fit_transform(df_ml[col])
            y = df_ml['cardio']
            X = df_ml.drop(['id', 'age', 'cardio'], axis=1, errors='ignore')
            continuous_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
            available_continuous = [col for col in continuous_columns if col in X.columns]
            if available_continuous:
                scaler = StandardScaler()
                X[available_continuous] = scaler.fit_transform(X[available_continuous])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            return X_train, X_test, y_train, y_test

        X_train_with, X_test_with, y_train_with, y_test_with = prepare_ml_data(data_with_outliers['data'])
        X_train_without, X_test_without, y_train_without, y_test_without = prepare_ml_data(data_without_outliers['data'])

        outlier_results, no_outlier_results = self.comparative_analyzer.run_comparative_analysis(
            {
                'X_train': X_train_with,
                'X_test': X_test_with,
                'y_train': y_train_with,
                'y_test': y_test_with,
                'feature_names': list(X_train_with.columns)
            },
            {
                'X_train': X_train_without,
                'X_test': X_test_without,
                'y_train': y_train_without,
                'y_test': y_test_without,
                'feature_names': list(X_train_without.columns)
            }
        )

        def compute_metrics(results_dict, y_test):
            metrics = {}
            for model_name, splits in results_dict['results'].items():
                y_proba = splits['test'].get('y_proba', None)
                if y_proba is None:
                    metrics[model_name] = {
                        'brier_score': None,
                        'log_loss': None,
                        'ece': None,
                        'calibration': None
                    }
                    continue
                try:
                    brier = brier_score_loss(y_test, y_proba)
                except Exception:
                    brier = None
                try:
                    ll = log_loss(y_test, y_proba)
                except Exception:
                    ll = None
                calib = self._compute_calibration_curve(y_test, y_proba, n_bins)
                metrics[model_name] = {
                    'brier_score': float(brier) if brier is not None else None,
                    'log_loss': float(ll) if ll is not None else None,
                    'ece': float(calib['ece']) if calib and calib.get('ece') is not None else None,
                    'calibration': calib
                }
            return metrics

        metrics_with = compute_metrics(outlier_results, y_test_with)
        metrics_without = compute_metrics(no_outlier_results, y_test_without)

        result = {
            'with_outliers': metrics_with,
            'without_outliers': metrics_without
        }
        return self._convert_numpy_types(result)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Risk skoruna göre risk seviyesini belirle."""
        if risk_score <= 0.2:
            return "Çok Düşük"
        elif risk_score <= 0.4:
            return "Düşük"
        elif risk_score <= 0.6:
            return "Orta"
        elif risk_score <= 0.8:
            return "Yüksek"
        else:
            return "Çok Yüksek"
    
    def _extract_feature_importance(self, importance_dfs: Dict) -> Dict[str, List[Dict]]:
        """Feature importance'ları çıkar."""
        feature_importance = {}
        
        for model_name, imp_df in importance_dfs.items():
            if imp_df is not None and not imp_df.empty:
                top_features = imp_df.head(10).to_dict('records')
                feature_importance[model_name] = top_features
        
        return feature_importance

    def _convert_numpy_types(self, data: Any) -> Any:
        """Veri yapısı içindeki numpy/pandas tiplerini Python yerel tiplerine dönüştür."""
        try:
            # numpy scalar tipleri
            if isinstance(data, np.generic):
                return data.item()
            # numpy int, float, bool
            if isinstance(data, (np.integer,)):
                return int(data)
            if isinstance(data, (np.floating,)):
                return float(data)
            if isinstance(data, (np.bool_,)):
                return bool(data)
            # numpy array
            if isinstance(data, np.ndarray):
                return data.tolist()
            # pandas timestamp
            if 'pandas' in sys.modules:
                import pandas as pd  # type: ignore
                if isinstance(data, pd.Timestamp):
                    return data.isoformat()
            # dict
            if isinstance(data, dict):
                return {self._convert_numpy_types(k): self._convert_numpy_types(v) for k, v in data.items()}
            # list/tuple
            if isinstance(data, list):
                return [self._convert_numpy_types(v) for v in data]
            if isinstance(data, tuple):
                return tuple(self._convert_numpy_types(v) for v in data)
            return data
        except Exception:
            # Güvenli tarafta kal: hatada orijinal veriyi döndür
            return data

    def _compute_calibration_curve(self, y_true, y_proba, n_bins: int = 10) -> Dict[str, Any]:
        """Kalibrasyon eğrisi (prob_true, prob_pred, bin_counts) ve ECE hesapla."""
        try:
            # Binleme
            y_true_arr = np.asarray(y_true)
            y_proba_arr = np.asarray(y_proba)
            # Güvenli sınır
            y_proba_arr = np.clip(y_proba_arr, 1e-12, 1 - 1e-12)
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            bin_ids = np.digitize(y_proba_arr, bins) - 1
            bin_ids = np.clip(bin_ids, 0, n_bins - 1)

            prob_true = []
            prob_pred = []
            counts = []
            ece = 0.0
            total = len(y_true_arr)
            for b in range(n_bins):
                mask = bin_ids == b
                if not np.any(mask):
                    prob_true.append(0.0)
                    prob_pred.append((bins[b] + bins[b+1]) / 2.0)
                    counts.append(0)
                    continue
                y_b = y_true_arr[mask]
                p_b = y_proba_arr[mask]
                acc_b = float(np.mean(y_b))
                conf_b = float(np.mean(p_b))
                prob_true.append(acc_b)
                prob_pred.append(conf_b)
                c = int(np.sum(mask))
                counts.append(c)
                ece += (c / total) * abs(acc_b - conf_b)

            return {
                'n_bins': int(n_bins),
                'prob_true': prob_true,
                'prob_pred': prob_pred,
                'bin_counts': counts,
                'ece': float(ece)
            }
        except Exception as e:
            print(f"Kalibrasyon eğrisi hesaplama hatası: {e}")
            return {
                'n_bins': int(n_bins),
                'prob_true': None,
                'prob_pred': None,
                'bin_counts': None,
                'ece': None
            }

    def _save_models_to_database(self, models_results: Dict[str, Any], data_type: str = "comparison") -> List[int]:
        """Modelleri veritabanına kaydet."""
        saved_ids = []
        db = SessionLocal()
        try:
            for model_name, results in models_results.items():
                # Model dosyasını kaydet
                model_path = self._save_model_file(results.get('model'), model_name, data_type)
                
                # Veritabanına kaydet
                model_record = ModelRecord(
                    model_name=f"{model_name}_{data_type}",
                    model_type="comparison",
                    accuracy=results.get('accuracy', 0.0),
                    precision=results.get('precision', 0.0),
                    recall=results.get('recall', 0.0),
                    f1_score=results.get('f1_score', 0.0),
                    roc_auc=results.get('roc_auc', 0.0),
                    brier_score=results.get('brier_score'),
                    log_loss=results.get('log_loss'),
                    ece_score=results.get('ece_score'),
                    model_path=model_path,
                    feature_names=json.dumps(results.get('feature_names', [])),
                    training_params=json.dumps(results.get('training_params', {})),
                    is_active=True
                )
                
                db.add(model_record)
                db.commit()
                db.refresh(model_record)
                saved_ids.append(model_record.id)
                
                print(f"✅ {model_name} modeli veritabanına kaydedildi (ID: {model_record.id})")
                
        except Exception as e:
            db.rollback()
            print(f"❌ Veritabanı kayıt hatası: {str(e)}")
            raise Exception(f"Veritabanı kayıt hatası: {str(e)}")
        finally:
            db.close()
        
        return saved_ids

    def _save_model_file(self, model, model_name: str, data_type: str) -> str:
        """Model dosyasını kaydet."""
        try:
            # Model klasörünü oluştur
            models_dir = project_root / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Dosya adı oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name.lower()}_{data_type}_{timestamp}.pkl"
            filepath = models_dir / filename
            
            # Modeli kaydet
            model_data = {
                'model': model,
                'model_name': model_name,
                'data_type': data_type,
                'timestamp': timestamp,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'model_type': 'comparison'
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Model dosyası kaydetme hatası: {str(e)}")
            return f"error_{model_name}_{data_type}_{timestamp}.pkl"

    def _get_model_type(self, model_name: str) -> str:
        """Model tipini belirle."""
        if "ensemble" in model_name.lower():
            return "ensemble"
        elif "tuned" in model_name.lower():
            return "tuned"
        else:
            return "base"

# Global service instance
model_comparison_service = ModelComparisonService()
