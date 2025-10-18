"""
Gerçek model verileri ile Brier Score, Log Loss ve Calibration Curve görselleştirmeleri oluştur.
Bu script mevcut model eğitimi pipeline'ını kullanarak gerçek verilerle görselleştirmeler üretir.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Mevcut modülleri import et
from brier_logloss_calibration_visualizer import ProbabilityMetricsVisualizer
from data.advanced_feature_engineering import AdvancedFeatureEngineer
from analysis.hyperparameter_tuning import HyperparameterTuner

class RealDataProbabilityMetricsGenerator:
    """Gerçek verilerle olasılık metrikleri görselleştirici."""
    
    def __init__(self, data_path=None):
        """
        Args:
            data_path: Veri dosyası yolu (None ise varsayılan yolu kullan)
        """
        self.data_path = data_path or (project_root / "data" / "cardiokaggle.csv")
        self.visualizer = ProbabilityMetricsVisualizer(style='bootstrap')
        self.models_dir = project_root / "models"
        
    def load_trained_models(self):
        """Eğitilmiş modelleri yükle."""
        models = {}
        
        if not self.models_dir.exists():
            print(f"Models dizini bulunamadı: {self.models_dir}")
            return models
        
        model_files = list(self.models_dir.glob("*.pkl"))
        
        for model_file in model_files:
            try:
                model_name = model_file.stem
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    models[model_name] = model_data
                print(f"Model yüklendi: {model_name}")
            except Exception as e:
                print(f"Model yüklenirken hata ({model_file}): {e}")
        
        return models
    
    def prepare_data(self):
        """Veriyi hazırla."""
        print("Veri hazırlanıyor...")
        
        if not self.data_path.exists():
            print(f"Veri dosyası bulunamadı: {self.data_path}")
            return None
        
        # Advanced Feature Engineering kullan
        advanced_fe = AdvancedFeatureEngineer()
        
        # Outlier'lı verilerle işleme
        data = advanced_fe.advanced_pipeline_with_outliers(str(self.data_path))
        
        if data is None:
            print("Veri işleme başarısız!")
            return None
        
        print(f"Veri hazırlandı:")
        print(f"- Eğitim seti: {data['X_train'].shape}")
        print(f"- Test seti: {data['X_test'].shape}")
        print(f"- Feature sayısı: {data['X_train'].shape[1]}")
        
        return data
    
    def train_new_models_if_needed(self, data):
        """Eğer model yoksa yeni modeller eğit."""
        models = self.load_trained_models()
        
        if len(models) < 3:  # En az 3 model olmalı
            print("Yeterli model bulunamadı, yeni modeller eğitiliyor...")
            
            tuner = HyperparameterTuner()
            best_models = tuner.tune_all_models(
                data['X_train'], data['y_train'], cv=3, n_jobs=-1
            )
            
            # Modelleri kaydet
            for model_name, model in best_models.items():
                model_path = self.models_dir / f"{model_name}_new.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Model kaydedildi: {model_path}")
            
            models = best_models
        
        return models
    
    def generate_probability_predictions(self, models, X_test):
        """Tüm modeller için olasılık tahminleri oluştur."""
        y_proba_dict = {}
        
        for model_name, model in models.items():
            try:
                # Model objesi mi yoksa dict mi kontrol et
                if isinstance(model, dict):
                    # Eğer dict ise, model'i al
                    model_obj = model.get('model', model)
                else:
                    model_obj = model
                
                # Olasılık tahminleri
                if hasattr(model_obj, 'predict_proba'):
                    y_proba = model_obj.predict_proba(X_test)[:, 1]
                    y_proba_dict[model_name] = y_proba
                    print(f"Olasılık tahminleri oluşturuldu: {model_name}")
                else:
                    print(f"Model olasılık tahmini desteklemiyor: {model_name}")
                    
            except Exception as e:
                print(f"Olasılık tahmini hatası ({model_name}): {e}")
        
        return y_proba_dict
    
    def create_outlier_comparison(self, data_with_outliers, data_without_outliers):
        """Outlier'lı ve outliersız veriler için karşılaştırma oluştur."""
        print("\n" + "="*60)
        print("OUTLIER KARŞILAŞTIRMASI - OLASILIK METRİKLERİ")
        print("="*60)
        
        # Her iki veri seti için modeller eğit
        models_with = self.train_new_models_if_needed(data_with_outliers)
        models_without = self.train_new_models_if_needed(data_without_outliers)
        
        # Olasılık tahminleri oluştur
        y_proba_with = self.generate_probability_predictions(
            models_with, data_with_outliers['X_test']
        )
        y_proba_without = self.generate_probability_predictions(
            models_without, data_without_outliers['X_test']
        )
        
        # Görselleştirmeler oluştur
        print("\nOutlier'lı veriler için görselleştirme oluşturuluyor...")
        metrics_with = self.visualizer.create_comprehensive_visualization(
            data_with_outliers['y_test'], y_proba_with,
            title_prefix="Outlier'lı Verilerle",
            save_path=str(project_root / "brier_logloss_calibration_with_outliers.png")
        )
        
        print("\nOutlier'sız veriler için görselleştirme oluşturuluyor...")
        metrics_without = self.visualizer.create_comprehensive_visualization(
            data_without_outliers['y_test'], y_proba_without,
            title_prefix="Outlier'sız Verilerle",
            save_path=str(project_root / "brier_logloss_calibration_without_outliers.png")
        )
        
        # Özet tabloları oluştur
        summary_with = self.visualizer.create_metrics_summary_table(
            metrics_with, 
            save_path=str(project_root / "metrics_summary_with_outliers.html")
        )
        
        summary_without = self.visualizer.create_metrics_summary_table(
            metrics_without,
            save_path=str(project_root / "metrics_summary_without_outliers.html")
        )
        
        return {
            'with_outliers': {'metrics': metrics_with, 'summary': summary_with},
            'without_outliers': {'metrics': metrics_without, 'summary': summary_without}
        }
    
    def create_combined_comparison(self, data_with_outliers, data_without_outliers):
        """Her iki veri setini birleştirerek karşılaştırmalı görselleştirme oluştur."""
        print("\n" + "="*60)
        print("BİRLEŞİK KARŞILAŞTIRMA - OLASILIK METRİKLERİ")
        print("="*60)
        
        # Her iki veri seti için modeller eğit
        models_with = self.train_new_models_if_needed(data_with_outliers)
        models_without = self.train_new_models_if_needed(data_without_outliers)
        
        # Olasılık tahminleri oluştur
        y_proba_with = self.generate_probability_predictions(
            models_with, data_with_outliers['X_test']
        )
        y_proba_without = self.generate_probability_predictions(
            models_without, data_without_outliers['X_test']
        )
        
        # Model isimlerini ayırt etmek için suffix ekle
        y_proba_combined = {}
        
        for model_name, y_proba in y_proba_with.items():
            y_proba_combined[f"{model_name} (Outlier'lı)"] = y_proba
        
        for model_name, y_proba in y_proba_without.items():
            y_proba_combined[f"{model_name} (Outlier'sız)"] = y_proba
        
        # Birleşik test seti oluştur
        y_test_combined = np.concatenate([
            data_with_outliers['y_test'],
            data_without_outliers['y_test']
        ])
        
        # Birleşik görselleştirme oluştur
        print("\nBirleşik karşılaştırma görselleştirmesi oluşturuluyor...")
        metrics_combined = self.visualizer.create_comprehensive_visualization(
            y_test_combined, y_proba_combined,
            title_prefix="Birleşik Karşılaştırma",
            save_path=str(project_root / "brier_logloss_calibration_combined.png")
        )
        
        # Özet tablo oluştur
        summary_combined = self.visualizer.create_metrics_summary_table(
            metrics_combined,
            save_path=str(project_root / "metrics_summary_combined.html")
        )
        
        return {
            'metrics': metrics_combined,
            'summary': summary_combined
        }
    
    def run_complete_analysis(self):
        """Tam analiz pipeline'ını çalıştır."""
        print("="*80)
        print("KALP KRİZİ RİSK TAHMİN MODELİ - OLASILIK METRİKLERİ ANALİZİ")
        print("="*80)
        
        # 1. Veriyi hazırla
        print("\n1. VERİ HAZIRLAMA")
        print("-" * 40)
        
        # Outlier'lı verilerle işleme
        print("Outlier'lı verilerle işleme...")
        data_with_outliers = self.prepare_data()
        
        # Outlier'sız verilerle işleme
        print("\nOutlier'sız verilerle işleme...")
        advanced_fe = AdvancedFeatureEngineer()
        data_without_outliers = advanced_fe.advanced_pipeline_without_outliers(str(self.data_path))
        
        if data_with_outliers is None or data_without_outliers is None:
            print("Veri hazırlama başarısız!")
            return None
        
        # 2. Outlier karşılaştırması
        print("\n2. OUTLIER KARŞILAŞTIRMASI")
        print("-" * 40)
        outlier_comparison = self.create_outlier_comparison(
            data_with_outliers, data_without_outliers
        )
        
        # 3. Birleşik karşılaştırma
        print("\n3. BİRLEŞİK KARŞILAŞTIRMA")
        print("-" * 40)
        combined_comparison = self.create_combined_comparison(
            data_with_outliers, data_without_outliers
        )
        
        # 4. Sonuçları yazdır
        print("\n4. SONUÇLAR")
        print("-" * 40)
        print("Oluşturulan dosyalar:")
        print("- brier_logloss_calibration_with_outliers.png")
        print("- brier_logloss_calibration_without_outliers.png")
        print("- brier_logloss_calibration_combined.png")
        print("- metrics_summary_with_outliers.html")
        print("- metrics_summary_without_outliers.html")
        print("- metrics_summary_combined.html")
        
        print("\n" + "="*80)
        print("ANALİZ TAMAMLANDI!")
        print("="*80)
        
        return {
            'outlier_comparison': outlier_comparison,
            'combined_comparison': combined_comparison
        }

def main():
    """Ana fonksiyon."""
    generator = RealDataProbabilityMetricsGenerator()
    results = generator.run_complete_analysis()
    
    if results:
        print("\nBaşarılı! Tüm görselleştirmeler oluşturuldu.")
    else:
        print("\nHata! Analiz tamamlanamadı.")

if __name__ == "__main__":
    main()
