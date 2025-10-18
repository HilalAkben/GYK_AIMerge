"""
Ensemble Methods Modülü - GPU Desteği ile
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
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

class EnsembleMethods:
    """
    Ensemble Methods sınıfı - GPU desteği ile
    """
    
    def __init__(self):
        """EnsembleMethods sınıfını başlat."""
        self.ensemble_models = {}
        self.ensemble_results = {}
        
    def create_base_models(self):
        """
        Temel modelleri oluştur.
        
        Returns:
            dict: Temel modeller
        """
        print("🔧 Temel modeller oluşturuluyor...")
        
        base_models = {}
        
        # Gradient Boosting
        from sklearn.ensemble import GradientBoostingClassifier
        base_models['gb'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # XGBoost (eğer mevcutsa)
        if XGBOOST_AVAILABLE:
            base_models['xgb'] = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        print(f"✅ {len(base_models)} temel model oluşturuldu")
        return base_models
    
    def create_voting_ensemble(self, base_models, voting='soft'):
        """
        Voting ensemble oluştur.
        
        Args:
            base_models: Temel modeller
            voting: Voting stratejisi ('soft' veya 'hard')
            
        Returns:
            VotingClassifier: Ensemble model
        """
        print(f"🗳️ {voting.capitalize()} voting ensemble oluşturuluyor...")
        
        estimators = [(name, model) for name, model in base_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        return ensemble
    
    def create_weighted_average_ensemble(self, base_models, X_train, y_train):
        """
        Ağırlıklı ortalama ensemble oluştur.
        
        Args:
            base_models: Temel modeller
            X_train: Eğitim verisi
            y_train: Eğitim etiketleri
            
        Returns:
            WeightedAverageEnsemble: Özel ensemble sınıfı
        """
        print("⚖️ Ağırlıklı ortalama ensemble oluşturuluyor...")
        
        # Modelleri eğit ve cross-validation skorlarını hesapla
        model_weights = {}
        
        for name, model in base_models.items():
            # Cross-validation ile ağırlık hesapla
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            avg_score = cv_scores.mean()
            model_weights[name] = avg_score
            print(f"  {name}: CV Score = {avg_score:.4f}")
        
        # Ağırlıkları normalize et
        total_weight = sum(model_weights.values())
        model_weights = {name: weight/total_weight for name, weight in model_weights.items()}
        
        print("Ağırlıklar:")
        for name, weight in model_weights.items():
            print(f"  {name}: {weight:.4f}")
        
        # Özel ensemble sınıfı oluştur
        ensemble = WeightedAverageEnsemble(base_models, model_weights)
        
        return ensemble
    
    def evaluate_ensemble(self, ensemble, X_train, X_test, y_train, y_test, ensemble_name):
        """
        Ensemble modeli değerlendir.
        
        Args:
            ensemble: Değerlendirilecek ensemble
            X_train: Eğitim verisi
            X_test: Test verisi
            y_train: Eğitim etiketleri
            y_test: Test etiketleri
            ensemble_name: Ensemble adı
            
        Returns:
            dict: Değerlendirme sonuçları
        """
        print(f"\n🔍 {ensemble_name} değerlendiriliyor...")
        
        # Modeli eğit
        ensemble.fit(X_train, y_train)
        
        # Tahminler
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }
        
        print(f"📊 {ensemble_name} Sonuçları:")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Precision: {precision:.4f}")
        print(f"  Test Recall: {recall:.4f}")
        print(f"  Test F1-Score: {f1:.4f}")
        print(f"  Test ROC AUC: {roc_auc:.4f}")
        print(f"  CV Accuracy: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        
        return results
    
    def run_ensemble_analysis(self, X_train, X_test, y_train, y_test):
        """
        Ensemble analizi çalıştır.
        
        Args:
            X_train: Eğitim verisi
            X_test: Test verisi
            y_train: Eğitim etiketleri
            y_test: Test etiketleri
            
        Returns:
            dict: Ensemble sonuçları
        """
        print("\n" + "="*60)
        print("ENSEMBLE METHODS ANALİZİ")
        print("="*60)
        
        # Temel modelleri oluştur
        base_models = self.create_base_models()
        
        results = {}
        
        # 1. Soft Voting Ensemble
        print("\n" + "="*50)
        print("1. SOFT VOTING ENSEMBLE")
        print("="*50)
        
        soft_voting = self.create_voting_ensemble(base_models, voting='soft')
        soft_results = self.evaluate_ensemble(
            soft_voting, X_train, X_test, y_train, y_test, "Soft Voting"
        )
        results['Soft Voting'] = soft_results
        self.ensemble_models['Soft Voting'] = soft_voting
        
        # 2. Hard Voting Ensemble
        print("\n" + "="*50)
        print("2. HARD VOTING ENSEMBLE")
        print("="*50)
        
        hard_voting = self.create_voting_ensemble(base_models, voting='hard')
        hard_results = self.evaluate_ensemble(
            hard_voting, X_train, X_test, y_train, y_test, "Hard Voting"
        )
        results['Hard Voting'] = hard_results
        self.ensemble_models['Hard Voting'] = hard_voting
        
        # 3. Weighted Average Ensemble
        print("\n" + "="*50)
        print("3. WEIGHTED AVERAGE ENSEMBLE")
        print("="*50)
        
        weighted_ensemble = self.create_weighted_average_ensemble(base_models, X_train, y_train)
        weighted_results = self.evaluate_ensemble(
            weighted_ensemble, X_train, X_test, y_train, y_test, "Weighted Average"
        )
        results['Weighted Average'] = weighted_results
        self.ensemble_models['Weighted Average'] = weighted_ensemble
        
        # En iyi modeli bul
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n🏆 En iyi ensemble: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Sonuçları DataFrame'e çevir
        results_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[name]['accuracy'] for name in results.keys()],
            'Precision': [results[name]['precision'] for name in results.keys()],
            'Recall': [results[name]['recall'] for name in results.keys()],
            'F1-Score': [results[name]['f1'] for name in results.keys()],
            'ROC AUC': [results[name]['roc_auc'] for name in results.keys()],
            'CV Accuracy': [results[name]['cv_mean'] for name in results.keys()]
        })
        
        self.ensemble_results = {
            'results': results,
            'results_df': results_df,
            'best_model_name': best_model_name,
            'best_accuracy': best_accuracy
        }
        
        return self.ensemble_results


class WeightedAverageEnsemble:
    """
    Ağırlıklı ortalama ensemble sınıfı
    """
    
    def __init__(self, base_models, weights):
        """
        WeightedAverageEnsemble sınıfını başlat.
        
        Args:
            base_models: Temel modeller
            weights: Model ağırlıkları
        """
        self.base_models = base_models
        self.weights = weights
        self.trained_models = {}
        
    def fit(self, X, y):
        """
        Modelleri eğit.
        
        Args:
            X: Eğitim verisi
            y: Eğitim etiketleri
        """
        for name, model in self.base_models.items():
            self.trained_models[name] = model.fit(X, y)
    
    def predict(self, X):
        """
        Tahmin yap.
        
        Args:
            X: Test verisi
            
        Returns:
            array: Tahminler
        """
        predictions = []
        
        for name, model in self.trained_models.items():
            pred = model.predict(X)
            weight = self.weights[name]
            predictions.append(pred * weight)
        
        # Ağırlıklı ortalama
        weighted_pred = np.sum(predictions, axis=0)
        
        # 0.5 eşiği ile sınıflandır
        return (weighted_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Olasılık tahminleri yap.
        
        Args:
            X: Test verisi
            
        Returns:
            array: Olasılık tahminleri
        """
        probabilities = []
        
        for name, model in self.trained_models.items():
            proba = model.predict_proba(X)[:, 1]  # Pozitif sınıf olasılığı
            weight = self.weights[name]
            probabilities.append(proba * weight)
        
        # Ağırlıklı ortalama
        weighted_proba = np.sum(probabilities, axis=0)
        
        # İki sınıf için olasılık matrisi oluştur
        proba_matrix = np.column_stack([1 - weighted_proba, weighted_proba])
        
        return proba_matrix
