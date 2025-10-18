"""
Veri analizi işlemleri için yardımcı sınıflar.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class DataAnalyzer:
    """
    Veri analizi işlemleri için ana sınıf.
    """
    
    def __init__(self):
        """DataAnalyzer sınıfını başlat."""
        pass
    
    def analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Eksik veri analizi yap.
        
        Args:
            df: Analiz edilecek DataFrame
            
        Returns:
            dict: Eksik veri analiz sonuçları
        """
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        missing_percentage = (total_missing / (df.shape[0] * df.shape[1])) * 100
        
        # Eksik veri olan kolonlar
        columns_with_missing = missing_counts[missing_counts > 0].index.tolist()
        
        result = {
            'has_missing': total_missing > 0,
            'total_missing': total_missing,
            'missing_percentage': missing_percentage,
            'missing_by_column': missing_counts.to_dict(),
            'columns_with_missing': columns_with_missing,
            'missing_percentage_by_column': (missing_counts / len(df) * 100).to_dict()
        }
        
        return result
    
    def analyze_numerical_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Nümerik özelliklerin analizi.
        
        Args:
            df: Nümerik özellikler içeren DataFrame
            
        Returns:
            dict: Nümerik özellik analiz sonuçları
        """
        # İstatistiksel özet
        summary = df.describe()
        
        # Her kolon için detaylı analiz
        column_analysis = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # Temel istatistikler
            stats = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'q1': col_data.quantile(0.25),
                'q3': col_data.quantile(0.75),
                'iqr': col_data.quantile(0.75) - col_data.quantile(0.25),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis()
            }
            
            # Aykırı değer analizi
            q1 = stats['q1']
            q3 = stats['q3']
            iqr = stats['iqr']
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            stats['outlier_count'] = len(outliers)
            stats['outlier_percentage'] = (len(outliers) / len(col_data)) * 100
            stats['lower_bound'] = lower_bound
            stats['upper_bound'] = upper_bound
            
            column_analysis[col] = stats
        
        result = {
            'summary': summary,
            'column_analysis': column_analysis,
            'total_columns': len(df.columns),
            'total_rows': len(df)
        }
        
        return result
    
    def analyze_categorical_features(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Kategorik özelliklerin analizi.
        
        Args:
            df: Kategorik özellikler içeren DataFrame
            
        Returns:
            dict: Kategorik özellik analiz sonuçları
        """
        results = {}
        
        for col in df.columns:
            col_data = df[col]
            
            # Temel istatistikler
            value_counts = col_data.value_counts()
            unique_count = col_data.nunique()
            most_common = value_counts.index[0] if len(value_counts) > 0 else None
            most_common_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
            
            # En az yaygın değer
            least_common = value_counts.index[-1] if len(value_counts) > 0 else None
            least_common_count = value_counts.iloc[-1] if len(value_counts) > 0 else 0
            
            # Entropi (çeşitlilik ölçüsü)
            probabilities = value_counts / len(col_data)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            # Eksik veri analizi
            missing_count = col_data.isnull().sum()
            missing_percentage = (missing_count / len(col_data)) * 100
            
            results[col] = {
                'unique_count': unique_count,
                'most_common': most_common,
                'most_common_count': most_common_count,
                'most_common_percentage': (most_common_count / len(col_data)) * 100,
                'least_common': least_common,
                'least_common_count': least_common_count,
                'least_common_percentage': (least_common_count / len(col_data)) * 100,
                'entropy': entropy,
                'missing_count': missing_count,
                'missing_percentage': missing_percentage,
                'value_counts': value_counts.to_dict(),
                'total_values': len(col_data)
            }
        
        return results
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> Dict[str, List[int]]:
        """
        Aykırı değerleri tespit et.
        
        Args:
            df: Analiz edilecek DataFrame
            method: Aykırı değer tespit yöntemi ('iqr' veya 'zscore')
            
        Returns:
            dict: Her kolon için aykırı değer indeksleri
        """
        outliers = {}
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                col_data = df[col].dropna()
                
                if method == 'iqr':
                    # IQR yöntemi
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    outlier_indices = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                    
                elif method == 'zscore':
                    # Z-score yöntemi
                    z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                    outlier_indices = df[col][z_scores > 3].index.tolist()
                
                outliers[col] = outlier_indices
        
        return outliers
    
    def analyze_correlations(self, df: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
        """
        Korelasyon analizi yap.
        
        Args:
            df: Analiz edilecek DataFrame
            method: Korelasyon yöntemi ('pearson', 'spearman', 'kendall')
            
        Returns:
            dict: Korelasyon analiz sonuçları
        """
        # Sadece nümerik kolonları al
        numerical_df = df.select_dtypes(include=[np.number])
        
        if numerical_df.empty:
            return {'error': 'Nümerik kolon bulunamadı'}
        
        # Korelasyon matrisi
        corr_matrix = numerical_df.corr(method=method)
        
        # Yüksek korelasyonlu özellik çiftleri
        high_corr_pairs = []
        threshold = 0.8  # Yüksek korelasyon eşiği
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # En yüksek korelasyonlu özellikler
        high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        result = {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_corr_pairs,
            'method': method,
            'threshold': threshold
        }
        
        return result
    
    def get_feature_importance_estimate(self, df: pd.DataFrame, target_col: str) -> Dict[str, float]:
        """
        Basit özellik önem tahmini (korelasyon bazlı).
        
        Args:
            df: Analiz edilecek DataFrame
            target_col: Hedef değişken kolonu
            
        Returns:
            dict: Özellik önem skorları
        """
        if target_col not in df.columns:
            return {}
        
        # Sadece nümerik kolonları al
        numerical_df = df.select_dtypes(include=[np.number])
        
        if target_col not in numerical_df.columns:
            return {}
        
        # Hedef değişkeni çıkar
        feature_cols = [col for col in numerical_df.columns if col != target_col]
        
        if not feature_cols:
            return {}
        
        # Korelasyon hesapla
        correlations = {}
        for col in feature_cols:
            corr = abs(numerical_df[col].corr(numerical_df[target_col]))
            correlations[col] = corr
        
        # Önem skorlarını sırala
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features) 