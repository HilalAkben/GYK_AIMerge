"""
Hyperparameter Tuning Modülü - GPU Desteği ile
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# GPU desteği için
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost GPU desteği aktif")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost yok, sadece GradientBoosting kullanılacak")

class HyperparameterTuner:
    """
    Hyperparameter Tuning sınıfı - GPU desteği ile
    """
    
    def __init__(self):
        """HyperparameterTuner sınıfını başlat."""
        self.best_models = {}
        self.tuning_results = {}
        
    def tune_gradient_boosting(self, X_train, y_train, cv=3, n_jobs=-1):
        """
        Gradient Boosting için hyperparameter tuning.
        
        Args:
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
            cv: Cross-validation fold sayısı
            n_jobs: Paralel iş sayısı
            
        Returns:
            GradientBoostingClassifier: En iyi model
        """
        print("🔧 Gradient Boosting Hyperparameter Tuning...")
        
        # Parametre grid'i
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Base model
        gb = GradientBoostingClassifier(random_state=42)
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=gb,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ En iyi parametreler: {grid_search.best_params_}")
        print(f"✅ En iyi skor: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def tune_xgboost(self, X_train, y_train, cv=3, n_jobs=-1):
        """
        XGBoost için hyperparameter tuning.
        
        Args:
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
            cv: Cross-validation fold sayısı
            n_jobs: Paralel iş sayısı
            
        Returns:
            XGBClassifier: En iyi model
        """
        if not XGBOOST_AVAILABLE:
            print("⚠️ XGBoost yok, GradientBoosting kullanılıyor")
            return self.tune_gradient_boosting(X_train, y_train, cv, n_jobs)
        
        print("🔧 XGBoost Hyperparameter Tuning...")
        
        # Parametre grid'i
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Base model
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ En iyi parametreler: {grid_search.best_params_}")
        print(f"✅ En iyi skor: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def tune_all_models(self, X_train, y_train, cv=3, n_jobs=-1):
        """
        Tüm modeller için hyperparameter tuning.
        
        Args:
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
            cv: Cross-validation fold sayısı
            n_jobs: Paralel iş sayısı
            
        Returns:
            dict: En iyi modeller
        """
        print("🚀 Tüm Modeller için Hyperparameter Tuning Başlatılıyor...")
        
        # Gradient Boosting
        print("\n" + "="*50)
        print("1. GRADIENT BOOSTING TUNING")
        print("="*50)
        gb_best = self.tune_gradient_boosting(X_train, y_train, cv, n_jobs)
        self.best_models['Gradient Boosting'] = gb_best
        
        # XGBoost
        print("\n" + "="*50)
        print("2. XGBOOST TUNING")
        print("="*50)
        xgb_best = self.tune_xgboost(X_train, y_train, cv, n_jobs)
        self.best_models['XGBoost'] = xgb_best
        
        print("\n✅ Tüm modeller için tuning tamamlandı!")
        return self.best_models
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Modeli değerlendir.
        
        Args:
            model: Değerlendirilecek model
            X_test: Test verisi
            y_test: Test etiketleri
            model_name: Model adı
            
        Returns:
            dict: Değerlendirme sonuçları
        """
        # Tahminler
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        print(f"\n📊 {model_name} Değerlendirme Sonuçları:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def evaluate_tuned_models(self, X_test, y_test):
        """
        Tune edilmiş modelleri değerlendir.
        
        Args:
            X_test: Test verisi
            y_test: Test etiketleri
            
        Returns:
            dict: Tüm modellerin değerlendirme sonuçları
        """
        print("\n" + "="*60)
        print("TUNE EDİLMİŞ MODELLERİN DEĞERLENDİRİLMESİ")
        print("="*60)
        
        results = {}
        
        for name, model in self.best_models.items():
            print(f"\n🔍 {name} değerlendiriliyor...")
            results[name] = self.evaluate_model(model, X_test, y_test, name)
        
        self.tuning_results = results
        return results
