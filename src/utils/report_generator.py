"""
Rapor oluşturma işlemleri için yardımcı sınıflar.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


class ReportGenerator:
    """
    Analiz raporları oluşturmak için ana sınıf.
    """
    
    def __init__(self):
        """ReportGenerator sınıfını başlat."""
        self.report_dir = Path(__file__).parent.parent.parent / "reports"
        self.report_dir.mkdir(exist_ok=True)
    
    def generate_analysis_report(self, df: pd.DataFrame, 
                               numerical_features: Optional[pd.DataFrame] = None,
                               categorical_features: Optional[pd.DataFrame] = None) -> None:
        """
        Kapsamlı analiz raporu oluştur.
        
        Args:
            df: Ana veri seti
            numerical_features: Nümerik özellikler DataFrame'i
            categorical_features: Kategorik özellikler DataFrame'i
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"data_analysis_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("KALP KRİZİ RİSK TAHMİN MODELİ - VERİ ANALİZ RAPORU\n")
            f.write("=" * 80 + "\n")
            f.write(f"Rapor Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 1. Genel Veri Seti Bilgileri
            f.write("1. GENEL VERİ SETİ BİLGİLERİ\n")
            f.write("-" * 50 + "\n")
            f.write(f"Veri Seti Boyutu: {df.shape[0]} satır x {df.shape[1]} kolon\n")
            f.write(f"Bellek Kullanımı: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
            f.write(f"Kolonlar: {', '.join(df.columns.tolist())}\n\n")
            
            # 2. Veri Türleri
            f.write("2. VERİ TÜRLERİ\n")
            f.write("-" * 50 + "\n")
            for col, dtype in df.dtypes.items():
                f.write(f"{col}: {dtype}\n")
            f.write("\n")
            
            # 3. Eksik Veri Analizi
            f.write("3. EKSİK VERİ ANALİZİ\n")
            f.write("-" * 50 + "\n")
            missing_counts = df.isnull().sum()
            total_missing = missing_counts.sum()
            
            if total_missing > 0:
                f.write(f"Toplam eksik değer sayısı: {total_missing}\n")
                f.write(f"Eksik değer yüzdesi: {(total_missing / (df.shape[0] * df.shape[1])) * 100:.2f}%\n\n")
                
                f.write("Kolon bazında eksik veriler:\n")
                for col, count in missing_counts.items():
                    if count > 0:
                        percentage = (count / len(df)) * 100
                        f.write(f"  {col}: {count} ({percentage:.2f}%)\n")
            else:
                f.write("Veri setinde eksik veri bulunmamaktadır.\n")
            f.write("\n")
            
            # 4. Nümerik Özellik Analizi
            if numerical_features is not None and not numerical_features.empty:
                f.write("4. NÜMERİK ÖZELLİK ANALİZİ\n")
                f.write("-" * 50 + "\n")
                
                summary = numerical_features.describe()
                f.write("İstatistiksel Özet:\n")
                f.write(str(summary) + "\n\n")
                
                # Her nümerik özellik için detaylı analiz
                for col in numerical_features.columns:
                    col_data = numerical_features[col]
                    
                    f.write(f"{col} Analizi:\n")
                    f.write(f"  Ortalama: {col_data.mean():.2f}\n")
                    f.write(f"  Medyan: {col_data.median():.2f}\n")
                    f.write(f"  Standart Sapma: {col_data.std():.2f}\n")
                    f.write(f"  Minimum: {col_data.min():.2f}\n")
                    f.write(f"  Maksimum: {col_data.max():.2f}\n")
                    f.write(f"  Çarpıklık: {col_data.skew():.2f}\n")
                    f.write(f"  Basıklık: {col_data.kurtosis():.2f}\n")
                    
                    # Aykırı değer analizi
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    f.write(f"  Aykırı Değer Sayısı: {len(outliers)} ({(len(outliers)/len(col_data)*100):.2f}%)\n")
                    f.write(f"  Aykırı Değer Sınırları: [{lower_bound:.2f}, {upper_bound:.2f}]\n\n")
            
            # 5. Kategorik Özellik Analizi
            if categorical_features is not None and not categorical_features.empty:
                f.write("5. KATEGORİK ÖZELLİK ANALİZİ\n")
                f.write("-" * 50 + "\n")
                
                for col in categorical_features.columns:
                    col_data = categorical_features[col]
                    value_counts = col_data.value_counts()
                    
                    f.write(f"{col} Analizi:\n")
                    f.write(f"  Benzersiz değer sayısı: {col_data.nunique()}\n")
                    f.write(f"  En yaygın değer: {value_counts.index[0]} ({value_counts.iloc[0]} kez)\n")
                    f.write(f"  En az yaygın değer: {value_counts.index[-1]} ({value_counts.iloc[-1]} kez)\n")
                    
                    # Entropi hesapla
                    probabilities = value_counts / len(col_data)
                    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                    f.write(f"  Entropi (çeşitlilik): {entropy:.2f}\n")
                    
                    f.write("  Değer dağılımı:\n")
                    for value, count in value_counts.items():
                        percentage = (count / len(col_data)) * 100
                        f.write(f"    {value}: {count} ({percentage:.2f}%)\n")
                    f.write("\n")
            
            # 6. Korelasyon Analizi
            if numerical_features is not None and not numerical_features.empty:
                f.write("6. KORELASYON ANALİZİ\n")
                f.write("-" * 50 + "\n")
                
                corr_matrix = numerical_features.corr()
                
                # Yüksek korelasyonlu özellik çiftleri
                high_corr_pairs = []
                threshold = 0.8
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) >= threshold:
                            high_corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': corr_value
                            })
                
                if high_corr_pairs:
                    f.write(f"Yüksek korelasyonlu özellik çiftleri (|r| >= {threshold}):\n")
                    for pair in high_corr_pairs:
                        f.write(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}\n")
                else:
                    f.write(f"Yüksek korelasyonlu özellik çifti bulunamadı (eşik: {threshold})\n")
                f.write("\n")
            
            # 7. Öneriler
            f.write("7. ÖNERİLER\n")
            f.write("-" * 50 + "\n")
            
            # Eksik veri önerileri
            if total_missing > 0:
                f.write("• Eksik veriler için uygun doldurma stratejileri belirlenmelidir.\n")
            
            # Aykırı değer önerileri
            if numerical_features is not None:
                outlier_cols = []
                for col in numerical_features.columns:
                    col_data = numerical_features[col]
                    q1 = col_data.quantile(0.25)
                    q3 = col_data.quantile(0.75)
                    iqr = q3 - q1
                    outliers = col_data[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]
                    if len(outliers) > 0:
                        outlier_cols.append(col)
                
                if outlier_cols:
                    f.write(f"• Aykırı değerler tespit edildi: {', '.join(outlier_cols)}\n")
                    f.write("  Bu değerler için uygun işleme stratejileri belirlenmelidir.\n")
            
            # Yüksek korelasyon önerileri
            high_corr_pairs = []
            if numerical_features is not None and not numerical_features.empty:
                corr_matrix = numerical_features.corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.8:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            if high_corr_pairs:
                f.write("• Yüksek korelasyonlu özellikler tespit edildi.\n")
                f.write("  Bu özelliklerden birini çıkararak çoklu doğrusallık sorunu önlenebilir.\n")
            
            f.write("• Veri ön işleme adımları tamamlandıktan sonra model geliştirme aşamasına geçilebilir.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("RAPOR SONU\n")
            f.write("=" * 80 + "\n")
        
        print(f"Analiz raporu oluşturuldu: {report_path}")
    
    def generate_summary_report(self, df: pd.DataFrame) -> None:
        """
        Özet rapor oluştur.
        
        Args:
            df: Analiz edilecek DataFrame
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"summary_report_{timestamp}.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("VERİ SETİ ÖZET RAPORU\n")
            f.write("=" * 50 + "\n")
            f.write(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Veri Seti Boyutu: {df.shape}\n")
            f.write(f"Kolon Sayısı: {len(df.columns)}\n")
            f.write(f"Satır Sayısı: {len(df)}\n")
            f.write(f"Eksik Veri Sayısı: {df.isnull().sum().sum()}\n")
            
            # Veri türleri
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            f.write(f"\nNümerik Özellikler: {len(numerical_cols)}\n")
            f.write(f"Kategorik Özellikler: {len(categorical_cols)}\n")
            
            if numerical_cols:
                f.write(f"\nNümerik Özellikler: {', '.join(numerical_cols)}\n")
            if categorical_cols:
                f.write(f"Kategorik Özellikler: {', '.join(categorical_cols)}\n")
        
        print(f"Özet rapor oluşturuldu: {report_path}")
    
    def save_dataframes_to_csv(self, df: pd.DataFrame, 
                              numerical_features: Optional[pd.DataFrame] = None,
                              categorical_features: Optional[pd.DataFrame] = None) -> None:
        """
        DataFrame'leri CSV olarak kaydet.
        
        Args:
            df: Ana DataFrame
            numerical_features: Nümerik özellikler DataFrame'i
            categorical_features: Kategorik özellikler DataFrame'i
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ana veri seti
        main_data_path = self.report_dir / f"main_dataset_{timestamp}.csv"
        df.to_csv(main_data_path, index=False)
        print(f"Ana veri seti kaydedildi: {main_data_path}")
        
        # Nümerik özellikler
        if numerical_features is not None and not numerical_features.empty:
            numerical_path = self.report_dir / f"numerical_features_{timestamp}.csv"
            numerical_features.to_csv(numerical_path, index=False)
            print(f"Nümerik özellikler kaydedildi: {numerical_path}")
        
        # Kategorik özellikler
        if categorical_features is not None and not categorical_features.empty:
            categorical_path = self.report_dir / f"categorical_features_{timestamp}.csv"
            categorical_features.to_csv(categorical_path, index=False)
            print(f"Kategorik özellikler kaydedildi: {categorical_path}") 