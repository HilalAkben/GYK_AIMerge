"""
Veri yükleme işlemleri için yardımcı sınıflar.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union


class DataLoader:
    """
    Veri yükleme işlemleri için ana sınıf.
    """
    
    def __init__(self):
        """DataLoader sınıfını başlat."""
        pass
    
    def load_csv(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        CSV dosyasını yükle.
        
        Args:
            file_path: CSV dosyasının yolu
            
        Returns:
            pd.DataFrame: Yüklenen veri seti
            
        Raises:
            FileNotFoundError: Dosya bulunamadığında
            pd.errors.EmptyDataError: Dosya boş olduğunda
        """
        try:
            # Önce noktalı virgül ile deneyin
            df = pd.read_csv(file_path, sep=';')
            print(f"Veri başarıyla yüklendi: {file_path} (noktalı virgül ayırıcı)")
            
            # Age verisini günden yıla çevir
            if 'age' in df.columns:
                df['age'] = df['age'] / 365.25  # Günü yıla çevir (365.25 gün/yıl)
                print("Age verisi günden yıla çevrildi.")
            
            return df
        except:
            try:
                # Virgül ile deneyin
                df = pd.read_csv(file_path, sep=',')
                print(f"Veri başarıyla yüklendi: {file_path} (virgül ayırıcı)")
                
                # Age verisini günden yıla çevir
                if 'age' in df.columns:
                    df['age'] = df['age'] / 365.25  # Günü yıla çevir (365.25 gün/yıl)
                    print("Age verisi günden yıla çevrildi.")
                
                return df
            except FileNotFoundError:
                raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
            except pd.errors.EmptyDataError:
                raise pd.errors.EmptyDataError(f"Dosya boş: {file_path}")
            except Exception as e:
                raise Exception(f"Veri yükleme hatası: {str(e)}")
    
    def save_csv(self, df: pd.DataFrame, file_path: Union[str, Path]) -> None:
        """
        DataFrame'i CSV dosyası olarak kaydet.
        
        Args:
            df: Kaydedilecek DataFrame
            file_path: Kayıt yolu
        """
        try:
            df.to_csv(file_path, index=False)
            print(f"Veri başarıyla kaydedildi: {file_path}")
        except Exception as e:
            raise Exception(f"Veri kaydetme hatası: {str(e)}")
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Veri seti hakkında genel bilgileri döndür.
        
        Args:
            df: Analiz edilecek DataFrame
            
        Returns:
            dict: Veri seti bilgileri
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }
        return info
    
    def display_data_info(self, df: pd.DataFrame) -> None:
        """
        Veri seti bilgilerini ekrana yazdır.
        
        Args:
            df: Bilgileri gösterilecek DataFrame
        """
        info = self.get_data_info(df)
        
        print(f"Veri Seti Boyutu: {info['shape']}")
        print(f"Bellek Kullanımı: {info['memory_usage'] / 1024 / 1024:.2f} MB")
        print(f"Kolon Sayısı: {len(info['columns'])}")
        print(f"Satır Sayısı: {info['shape'][0]}")
        
        print("\nKolon Türleri:")
        for col, dtype in info['dtypes'].items():
            print(f"  {col}: {dtype}")
            
        print("\nBenzersiz Değer Sayıları:")
        for col, unique_count in info['unique_counts'].items():
            print(f"  {col}: {unique_count}")
            
        print("\nEksik Veri Sayıları:")
        for col, null_count in info['null_counts'].items():
            if null_count > 0:
                print(f"  {col}: {null_count}")
            else:
                print(f"  {col}: Eksik veri yok")
