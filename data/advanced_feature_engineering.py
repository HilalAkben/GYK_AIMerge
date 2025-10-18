"""
Gelişmiş Feature Engineering Modülü - GPU Desteği ile
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# GPU desteği için
GPU_AVAILABLE = False
try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
    print("✅ GPU desteği aktif (CuPy)")
except (ImportError, ModuleNotFoundError):
    GPU_AVAILABLE = False
    print("⚠️ GPU desteği yok, CPU kullanılıyor")

class AdvancedFeatureEngineer:
    """
    Gelişmiş Feature Engineering sınıfı - GPU desteği ile
    """
    
    def __init__(self):
        """AdvancedFeatureEngineer sınıfını başlat."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
    def load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        """
        Veriyi yükle ve temel ön işleme yap.
        
        Args:
            data_path: Veri dosyasının yolu
            
        Returns:
            pd.DataFrame: Ön işlenmiş veri
        """
        print("📊 Veri yükleniyor...")
        
        # Veriyi yükle
        try:
            df = pd.read_csv(data_path, sep=';')
        except:
            df = pd.read_csv(data_path, sep=',')
        
        print(f"✅ Veri yüklendi: {df.shape}")
        
        # Age'i günden yıla çevir
        if 'age' in df.columns:
            df['age'] = df['age'] / 365.25
            print("✅ Age günden yıla çevrildi")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelişmiş özellikler oluştur.
        
        Args:
            df: Ham veri
            
        Returns:
            pd.DataFrame: Gelişmiş özelliklerle veri
        """
        print("🔧 Gelişmiş özellikler oluşturuluyor...")
        
        df_advanced = df.copy()
        
        # BMI hesapla
        if 'height' in df.columns and 'weight' in df.columns:
            df_advanced['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
            print("✅ BMI hesaplandı")
        
        # Tansiyon farkı
        if 'ap_hi' in df.columns and 'ap_lo' in df.columns:
            df_advanced['bp_diff'] = df['ap_hi'] - df['ap_lo']
            df_advanced['bp_ratio'] = df['ap_hi'] / df['ap_lo']
            print("✅ Tansiyon farkı ve oranı hesaplandı")
        
        # Yaş grupları
        if 'age' in df.columns:
            df_advanced['age_group'] = pd.cut(df['age'], 
                                            bins=[0, 30, 45, 60, 100], 
                                            labels=['Genç', 'Orta', 'Yaşlı', 'Çok Yaşlı'])
            print("✅ Yaş grupları oluşturuldu")
        
        # Kolesterol ve glikoz etkileşimi
        if 'cholesterol' in df.columns and 'gluc' in df.columns:
            df_advanced['chol_gluc_interaction'] = df['cholesterol'] * df['gluc']
            print("✅ Kolesterol-Glikoz etkileşimi oluşturuldu")
        
        # Risk skorları
        risk_factors = []
        if 'smoke' in df.columns:
            risk_factors.append(df['smoke'])
        if 'alco' in df.columns:
            risk_factors.append(df['alco'])
        if 'active' in df.columns:
            risk_factors.append(1 - df['active'])  # Aktif olmamak risk faktörü
        
        if risk_factors:
            df_advanced['risk_score'] = sum(risk_factors)
            print("✅ Risk skoru hesaplandı")
        
        return df_advanced
    
    def handle_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Kategorik özellikleri işle.
        
        Args:
            df: Veri
            
        Returns:
            pd.DataFrame: Kategorik özellikleri işlenmiş veri
        """
        print("🏷️ Kategorik özellikler işleniyor...")
        
        categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        available_categorical = [col for col in categorical_cols if col in df.columns]
        
        df_encoded = df.copy()
        
        for col in available_categorical:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"✅ {col} encode edildi")
        
        return df_encoded
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Eksik değerleri işle.
        
        Args:
            df: Veri
            
        Returns:
            pd.DataFrame: Eksik değerleri işlenmiş veri
        """
        print("🔍 Eksik değerler kontrol ediliyor...")
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"⚠️ Eksik değerler bulundu: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Sayısal kolonlar için median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            df[numerical_cols] = self.imputer.fit_transform(df[numerical_cols])
            
            # Kategorik kolonlar için mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            
            print("✅ Eksik değerler dolduruldu")
        else:
            print("✅ Eksik değer yok")
        
        return df
    
    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Özellikleri ölçeklendir.
        
        Args:
            df: Veri
            
        Returns:
            pd.DataFrame: Ölçeklendirilmiş veri
        """
        print("📏 Özellikler ölçeklendiriliyor...")
        
        # Sayısal kolonları seç (target hariç)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if 'cardio' in numerical_cols:
            numerical_cols = numerical_cols.drop('cardio')
        
        if len(numerical_cols) > 0:
            df_scaled = df.copy()
            df_scaled[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            print(f"✅ {len(numerical_cols)} özellik ölçeklendirildi")
            return df_scaled
        
        return df
    
    def prepare_final_data(self, df: pd.DataFrame) -> dict:
        """
        Son veri hazırlığını yap.
        
        Args:
            df: İşlenmiş veri
            
        Returns:
            dict: Eğitim ve test verileri
        """
        print("🎯 Son veri hazırlığı...")
        
        # Target değişkeni ayır
        if 'cardio' in df.columns:
            X = df.drop('cardio', axis=1)
            y = df['cardio']
        else:
            raise ValueError("Target değişkeni 'cardio' bulunamadı!")
        
        # Feature isimlerini kaydet
        self.feature_names = list(X.columns)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Veri bölündü: Train {X_train.shape}, Test {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
    
    def advanced_pipeline_with_outliers(self, data_path: str) -> dict:
        """
        Outlier'lı verilerle gelişmiş pipeline.
        
        Args:
            data_path: Veri dosyasının yolu
            
        Returns:
            dict: İşlenmiş veri
        """
        print("🚀 Gelişmiş Pipeline Başlatılıyor (Outlier'lı Verilerle)...")
        
        try:
            # 1. Veri yükleme
            df = self.load_and_preprocess_data(data_path)
            
            # 2. Gelişmiş özellikler
            df = self.create_advanced_features(df)
            
            # 3. Kategorik özellikler
            df = self.handle_categorical_features(df)
            
            # 4. Eksik değerler
            df = self.handle_missing_values(df)
            
            # 5. Ölçeklendirme
            df = self.scale_features(df)
            
            # 6. Son hazırlık
            data = self.prepare_final_data(df)
            
            print("🎉 Gelişmiş Pipeline Tamamlandı!")
            return data
            
        except Exception as e:
            print(f"❌ Pipeline hatası: {str(e)}")
            return None
