"""
Risk Tahmin Servisi
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from api.core.database import ModelRecord, PredictionRecord, SessionLocal, PatientRiskRecord, Appointment, User
from sklearn.preprocessing import StandardScaler, LabelEncoder

class PredictionService:
    """Risk tahmin servisi."""
    
    def __init__(self):
        self.models_dir = project_root / "models"
    
    async def predict_cardio_risk(
        self, 
        model_id: int,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hastanın kardiyovasküler riskini tahmin et.
        
        Args:
            model_id: Kullanılacak model ID'si
            patient_data: Hasta verileri
            
        Returns:
            Risk tahmini sonuçları
        """
        try:
            # 1. Modeli veritabanından al
            model_record = self._get_model_from_database(model_id)
            if not model_record:
                raise Exception(f"Model ID {model_id} bulunamadı!")
            
            # 2. Modeli yükle
            trained_model = self._load_model(model_record.model_path)
            if not trained_model:
                raise Exception("Model yüklenemedi!")
            
            # 3. Hasta verilerini hazırla
            prepared_data = self._prepare_patient_data(
                patient_data, 
                model_record.feature_names,
                trained_model.get('feature_names', [])
            )
            
            # 4. Tahmin yap
            prediction_result = self._make_prediction(trained_model['model'], prepared_data)
            
            # 5. Risk skorunu hesapla (model olasılığı + kural tabanlı risk)
            risk_score = self._calculate_risk_score(
                prediction_result['probability'],
                model_record.f1_score,
                patient_data
            )
            
            # 6. Risk seviyesini belirle
            risk_level = self._determine_risk_level(risk_score)
            
            # 7. Önerileri oluştur
            recommendations = self._generate_recommendations(
                patient_data, risk_score, risk_level
            )
            
            # 8. Tahmini veritabanına kaydet
            prediction_record_id = self._save_prediction_to_database(
                model_id, patient_data, prediction_result, risk_score
            )
            
            # Final sonuçlar
            result = {
                'prediction_summary': {
                    'model_name': model_record.model_name,
                    'model_accuracy': float(model_record.accuracy) if model_record.accuracy is not None else None,
                    'prediction': int(prediction_result['prediction']),
                    'probability': float(prediction_result['probability']),
                    'risk_score': float(risk_score),
                    'risk_level': risk_level,
                    'prediction_id': prediction_record_id
                },
                'patient_analysis': {
                    'input_data': patient_data,
                    'risk_factors': self._identify_risk_factors(patient_data),
                    'protective_factors': self._identify_protective_factors(patient_data)
                },
                'recommendations': recommendations,
                'model_info': {
                    'model_type': model_record.model_type,
                    'training_date': model_record.created_at.isoformat(),
                    'feature_count': (len(json.loads(model_record.feature_names)) if isinstance(model_record.feature_names, str) and model_record.feature_names else (len(model_record.feature_names) if model_record.feature_names else 0))
                },
                'prediction_timestamp': datetime.utcnow().isoformat()
            }
            
            # Numpy/Pandas tiplerini native Python tiplere dönüştür
            result = self._convert_numpy_types(result)
            return result
            
        except Exception as e:
            raise Exception(f"Tahmin hatası: {str(e)}")

    async def auto_select_and_predict(
        self,
        data_path: str,
        patient_data: Dict[str, Any],
        weight_brier: float = 0.5,
        weight_logloss: float = 0.5,
    ) -> Dict[str, Any]:
        """Veritabanındaki en iyi modeli seç, risk tahmini yap ve hibrit öncelik puanı hesapla."""
        db = SessionLocal()
        try:
            # Veritabanından en iyi performanslı aktif modeli seç (F1 skoruna göre)
            model_record = db.query(ModelRecord).filter(
                ModelRecord.is_active == True
            ).order_by(ModelRecord.f1_score.desc()).first()
            
            if not model_record:
                raise Exception("Aktif model bulunamadı")

            # Modeli yükle
            trained_model = self._load_model(model_record.model_path)
            if not trained_model:
                raise Exception(f"Model yüklenemedi: {model_record.model_path}")

            # Hasta verilerini hazırla
            prepared = self._prepare_patient_data(
                patient_data,
                model_record.feature_names,
                trained_model.get('feature_names', []) if trained_model else []
            )
            
            # Tahmin yap
            pred = self._make_prediction(trained_model['model'], prepared)
            risk_score = self._calculate_risk_score(pred['probability'], model_record.f1_score, patient_data)

            # Hibrit ağırlıklı faktör skoru hesapla
            weighted_factors_score = self._compute_weighted_factor_score(
                trained_model.get('evaluation_results', {}).get('feature_importance') if trained_model else None,
                patient_data
            )
            priority_score = 0.6 * float(risk_score) + 0.4 * float(weighted_factors_score)

            # kayıt
            self._save_patient_risk_record(
                db, patient_data, model_record.model_name,
                float(risk_score), float(weighted_factors_score), float(priority_score)
            )

            return {
                'best_model_name': model_record.model_name,
                'risk_score': float(risk_score),
                'priority_score': float(priority_score),
                'probability': float(pred['probability']),
                'prediction': int(pred['prediction']),
                'weighted_factors_score': float(weighted_factors_score),
            }
        finally:
            db.close()

    def _compute_weighted_factor_score(self, feature_importance: Optional[Dict[str, Any]], patient_data: Dict[str, Any]) -> float:
        clinical_weights = {
            'age': 0.9,
            'ap_hi': 0.85,
            'cholesterol': 0.9,
            'gluc': 0.85,
            'smoke': 0.8,
            'alco': 0.6,
            'active': 0.4,
        }

        importance_map: Dict[str, float] = {}
        if isinstance(feature_importance, dict):
            val0 = next(iter(feature_importance.values()), None)
            if isinstance(val0, list):
                for items in feature_importance.values():
                    for it in items:
                        f = it.get('feature')
                        v = float(it.get('importance', 0.0))
                        if f:
                            importance_map[f] = max(importance_map.get(f, 0.0), v)
            else:
                for k, v in feature_importance.items():
                    try:
                        importance_map[k] = float(v)
                    except:
                        pass

        score = 0.0
        total_weight = 0.0
        for key, clin_w in clinical_weights.items():
            model_w = importance_map.get(key, 0.0)
            weight = clin_w * model_w
            val = float(patient_data.get(key, 0) or 0)
            if key in ['smoke', 'alco', 'active']:
                val = 1.0 if int(val) == 1 else 0.0
            score += weight * val
            total_weight += max(weight, 1e-6)
        return float(score / total_weight) if total_weight > 0 else 0.0

    def _save_patient_risk_record(
        self,
        db,
        patient_data: Dict[str, Any],
        best_model_name: str,
        risk_score: float,
        weighted_factors_score: float,
        priority_score: float,
    ) -> int:
        rec = PatientRiskRecord(
            age=int(patient_data['age']),
            gender=int(patient_data['gender']),
            height=float(patient_data['height']),
            weight=float(patient_data['weight']),
            ap_hi=int(patient_data['ap_hi']),
            ap_lo=int(patient_data['ap_lo']),
            cholesterol=int(patient_data['cholesterol']),
            gluc=int(patient_data['gluc']),
            smoke=bool(patient_data['smoke']),
            alco=bool(patient_data['alco']),
            active=bool(patient_data['active']),
            best_model_name=best_model_name,
            risk_score=risk_score,
            weighted_factors_score=weighted_factors_score,
            priority_score=priority_score,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec.id
    
    def _get_model_from_database(self, model_id: int) -> Optional[ModelRecord]:
        """Modeli veritabanından al."""
        db = SessionLocal()
        try:
            # Model ID 0 ise en iyi modeli seç
            if model_id == 0:
                model_record = db.query(ModelRecord).filter(
                    ModelRecord.is_active == True
                ).order_by(ModelRecord.f1_score.desc()).first()
            else:
                model_record = db.query(ModelRecord).filter(
                    ModelRecord.id == model_id,
                    ModelRecord.is_active == True
                ).first()
            return model_record
        except Exception as e:
            print(f"Veritabanı hatası: {e}")
            return None
        finally:
            db.close()
    
    def _load_model(self, model_path: str):
        """Modeli dosyadan yükle."""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return None
    
    def _prepare_patient_data(self, patient_data: Dict, model_feature_names: List, trained_feature_names: List):
        """Hasta verilerini model için hazırla."""
        # model_feature_names JSON string olabilir; listeye çevir
        parsed_feature_names = None
        if isinstance(model_feature_names, str):
            try:
                parsed_feature_names = json.loads(model_feature_names)
            except Exception:
                parsed_feature_names = None
        # Önce model kaydındaki feature names'i kullan
        feature_names = parsed_feature_names if parsed_feature_names else (model_feature_names if model_feature_names else trained_feature_names)
        
        if not feature_names:
            raise Exception("Model feature names bulunamadı!")
        
        # Veri doğrulama
        required_fields = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        missing_fields = [field for field in required_fields if field not in patient_data]
        
        if missing_fields:
            raise Exception(f"Eksik alanlar: {', '.join(missing_fields)}")
        
        # Veri tiplerini kontrol et ve dönüştür
        processed_data = {}
        
        # Yaş dönüşümü (gün -> yıl)
        if 'age' in patient_data:
            age_days = patient_data['age']
            if age_days > 1000:  # Gün formatında ise
                processed_data['age_years'] = int(age_days / 365)
            else:  # Yıl formatında ise
                processed_data['age_years'] = int(age_days)
        
        # Diğer sayısal alanlar
        numeric_fields = ['height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
        for field in numeric_fields:
            if field in patient_data:
                processed_data[field] = float(patient_data[field])
        
        # Kategorik alanlar
        categorical_fields = ['gender', 'smoke', 'alco', 'active']
        for field in categorical_fields:
            if field in patient_data:
                processed_data[field] = int(patient_data[field])
        
        # DataFrame oluştur
        df = pd.DataFrame([processed_data])
        
        # Feature names'e göre sırala ve eksikleri doldur
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Eksik feature'ları 0 ile doldur
        
        # Sadece model feature'larını al
        df = df[feature_names]
        
        return df
    
    def _make_prediction(self, model, prepared_data):
        """Model ile tahmin yap."""
        try:
            # Tahmin
            prediction = model.predict(prepared_data)[0]
            
            # Olasılık (eğer model destekliyorsa)
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(prepared_data)[0][1]  # Pozitif sınıf olasılığı
            else:
                probability = float(prediction)  # Binary prediction için
            
            return {
                'prediction': prediction,
                'probability': probability
            }
            
        except Exception as e:
            raise Exception(f"Tahmin hatası: {str(e)}")
    
    def _calculate_risk_score(self, probability: float, model_f1_score: float, patient_data: Dict[str, Any]) -> float:
        """
        Risk skorunu hesapla.
        
        Args:
            probability: Model tahmin olasılığı
            model_f1_score: Model F1 skoru
            patient_data: Hasta verileri (kural tabanlı risk için)
            
        Returns:
            Risk skoru (0-1 arası)
        """
        # 1) Model olasılığını güvenilirliğe göre ayarla
        base_risk = float(probability)
        model_reliability = float(model_f1_score or 0.0)
        adjusted_prob_risk = base_risk * (0.7 + 0.3 * model_reliability)

        # 2) Kural tabanlı risk (aşırı değerleri yakala)
        rule_risk = self._compute_rule_based_risk(patient_data)

        # 3) Nihai risk = iki yöntemin maksimumu (aşırı klinik değerler baskın)
        risk_score = max(adjusted_prob_risk, rule_risk)

        # 0-1 aralığına sıkıştır
        return max(0.0, min(1.0, risk_score))

    def _compute_rule_based_risk(self, patient_data: Dict[str, Any]) -> float:
        """Klinik eşiğe dayalı basit kural tabanlı risk (0-1). Aşırı değerler riski artırır."""
        try:
            score = 0.0
            # Yaş
            age = patient_data.get('age', 0)
            age_years = age / 365 if age and age > 1000 else age
            if age_years >= 70:
                score += 0.25
            elif age_years >= 60:
                score += 0.15
            elif age_years >= 50:
                score += 0.1

            # Kan basıncı
            ap_hi = float(patient_data.get('ap_hi', 0) or 0)
            ap_lo = float(patient_data.get('ap_lo', 0) or 0)
            if ap_hi >= 180 or ap_lo >= 110:
                score += 0.35
            elif ap_hi >= 160 or ap_lo >= 100:
                score += 0.25
            elif ap_hi >= 140 or ap_lo >= 90:
                score += 0.15

            # Kolesterol
            chol = int(patient_data.get('cholesterol', 1) or 1)
            if chol >= 3:
                score += 0.15
            elif chol == 2:
                score += 0.1

            # Glukoz
            gluc = int(patient_data.get('gluc', 1) or 1)
            if gluc >= 3:
                score += 0.15
            elif gluc == 2:
                score += 0.1

            # BMI
            height = float(patient_data.get('height', 0) or 0)
            weight = float(patient_data.get('weight', 0) or 0)
            if height > 0 and weight > 0:
                bmi = weight / ((height / 100.0) ** 2)
                if bmi >= 35:
                    score += 0.25
                elif bmi >= 30:
                    score += 0.15
                elif bmi >= 25:
                    score += 0.05

            # Yaşam tarzı
            smoke = int(patient_data.get('smoke', 0) or 0)
            alco = int(patient_data.get('alco', 0) or 0)
            active = int(patient_data.get('active', 1) or 1)
            if smoke == 1:
                score += 0.1
            if alco == 1:
                score += 0.05
            if active == 0:
                score += 0.05

            # Üst sınır
            return min(score, 1.0)
        except Exception:
            return 0.0
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Risk seviyesini belirle."""
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
    
    def _identify_risk_factors(self, patient_data: Dict) -> List[str]:
        """Risk faktörlerini belirle."""
        risk_factors = []
        
        # Yaş faktörü
        age = patient_data.get('age', 0)
        if age > 1000:  # Gün formatında
            age_years = age / 365
        else:
            age_years = age
            
        if age_years > 65:
            risk_factors.append("İleri yaş")
        elif age_years > 50:
            risk_factors.append("Orta yaş")
        
        # Cinsiyet faktörü
        gender = patient_data.get('gender', 1)
        if gender == 1:  # Erkek
            risk_factors.append("Erkek cinsiyet")
        
        # Tansiyon faktörü
        ap_hi = patient_data.get('ap_hi', 0)
        ap_lo = patient_data.get('ap_lo', 0)
        
        if ap_hi >= 140 or ap_lo >= 90:
            risk_factors.append("Yüksek tansiyon")
        elif ap_hi >= 130 or ap_lo >= 80:
            risk_factors.append("Pre-hipertansiyon")
        
        # Kolesterol faktörü
        cholesterol = patient_data.get('cholesterol', 1)
        if cholesterol >= 3:
            risk_factors.append("Yüksek kolesterol")
        
        # Glukoz faktörü
        gluc = patient_data.get('gluc', 1)
        if gluc >= 3:
            risk_factors.append("Yüksek glukoz")
        
        # Sigara faktörü
        smoke = patient_data.get('smoke', 0)
        if smoke == 1:
            risk_factors.append("Sigara kullanımı")
        
        # Alkol faktörü
        alco = patient_data.get('alco', 0)
        if alco == 1:
            risk_factors.append("Alkol kullanımı")
        
        # Hareketsizlik faktörü
        active = patient_data.get('active', 0)
        if active == 0:
            risk_factors.append("Hareketsiz yaşam")
        
        # BMI faktörü
        height = patient_data.get('height', 0)
        weight = patient_data.get('weight', 0)
        if height > 0 and weight > 0:
            bmi = weight / ((height / 100) ** 2)
            if bmi >= 30:
                risk_factors.append("Obezite")
            elif bmi >= 25:
                risk_factors.append("Fazla kilolu")
        
        return risk_factors
    
    def _identify_protective_factors(self, patient_data: Dict) -> List[str]:
        """Koruyucu faktörleri belirle."""
        protective_factors = []
        
        # Aktif yaşam
        active = patient_data.get('active', 0)
        if active == 1:
            protective_factors.append("Aktif yaşam tarzı")
        
        # Sigara kullanmama
        smoke = patient_data.get('smoke', 0)
        if smoke == 0:
            protective_factors.append("Sigara kullanmama")
        
        # Alkol kullanmama
        alco = patient_data.get('alco', 0)
        if alco == 0:
            protective_factors.append("Alkol kullanmama")
        
        # Normal tansiyon
        ap_hi = patient_data.get('ap_hi', 0)
        ap_lo = patient_data.get('ap_lo', 0)
        if ap_hi < 120 and ap_lo < 80:
            protective_factors.append("Normal tansiyon")
        
        # Normal kolesterol
        cholesterol = patient_data.get('cholesterol', 1)
        if cholesterol == 1:
            protective_factors.append("Normal kolesterol")
        
        # Normal glukoz
        gluc = patient_data.get('gluc', 1)
        if gluc == 1:
            protective_factors.append("Normal glukoz")
        
        # Normal BMI
        height = patient_data.get('height', 0)
        weight = patient_data.get('weight', 0)
        if height > 0 and weight > 0:
            bmi = weight / ((height / 100) ** 2)
            if 18.5 <= bmi < 25:
                protective_factors.append("Normal kilo")
        
        return protective_factors
    
    def _generate_recommendations(self, patient_data: Dict, risk_score: float, risk_level: str) -> Dict[str, List[str]]:
        """Öneriler oluştur."""
        recommendations = {
            'lifestyle': [],
            'medical': [],
            'monitoring': []
        }
        
        # Risk seviyesine göre genel öneriler
        if risk_level in ["Yüksek", "Çok Yüksek"]:
            recommendations['lifestyle'].extend([
                "Düzenli egzersiz yapın (haftada en az 150 dakika)",
                "Sağlıklı beslenme programı uygulayın",
                "Stres yönetimi teknikleri öğrenin",
                "Yeterli uyku alın (7-8 saat)"
            ])
            
            recommendations['medical'].extend([
                "Kardiyoloji uzmanına başvurun",
                "Düzenli kan basıncı takibi yapın",
                "Lipid profili kontrolü yaptırın",
                "Diyabet taraması yaptırın"
            ])
            
            recommendations['monitoring'].extend([
                "Günlük kan basıncı ölçümü",
                "Aylık kilo takibi",
                "3 ayda bir kan testleri",
                "Yıllık kardiyak değerlendirme"
            ])
            
        elif risk_level == "Orta":
            recommendations['lifestyle'].extend([
                "Haftada en az 3 gün egzersiz yapın",
                "Meyve ve sebze tüketimini artırın",
                "Tuz tüketimini azaltın",
                "Sigara ve alkol kullanımından kaçının"
            ])
            
            recommendations['medical'].extend([
                "Aile hekiminize düzenli kontrole gidin",
                "Kan basıncınızı takip edin",
                "Kolesterol seviyelerinizi kontrol ettirin"
            ])
            
            recommendations['monitoring'].extend([
                "Haftalık kan basıncı ölçümü",
                "Aylık kilo takibi",
                "6 ayda bir kan testleri"
            ])
            
        else:  # Düşük risk
            recommendations['lifestyle'].extend([
                "Mevcut sağlıklı yaşam tarzınızı sürdürün",
                "Düzenli fiziksel aktivite yapın",
                "Sağlıklı beslenme alışkanlıklarınızı koruyun"
            ])
            
            recommendations['medical'].extend([
                "Yıllık sağlık kontrolü yaptırın",
                "Aile geçmişinizi takip edin"
            ])
            
            recommendations['monitoring'].extend([
                "Yıllık kan basıncı kontrolü",
                "Yıllık kan testleri"
            ])
        
        # Spesifik risk faktörlerine göre öneriler
        risk_factors = self._identify_risk_factors(patient_data)
        
        if "Yüksek tansiyon" in risk_factors:
            recommendations['medical'].append("Hipertansiyon tedavisi için doktora başvurun")
            recommendations['lifestyle'].append("Düşük sodyum diyeti uygulayın")
        
        if "Yüksek kolesterol" in risk_factors:
            recommendations['medical'].append("Kolesterol düşürücü tedavi için doktora başvurun")
            recommendations['lifestyle'].append("Düşük kolesterol diyeti uygulayın")
        
        if "Sigara kullanımı" in risk_factors:
            recommendations['lifestyle'].append("Sigara bırakma programına katılın")
            recommendations['medical'].append("Sigara bırakma danışmanlığı alın")
        
        if "Obezite" in risk_factors:
            recommendations['lifestyle'].append("Kilo verme programına katılın")
            recommendations['medical'].append("Beslenme uzmanından yardım alın")
        
        return recommendations
    
    def _save_prediction_to_database(self, model_id: int, patient_data: Dict, prediction_result: Dict, risk_score: float) -> int:
        """Tahmini veritabanına kaydet."""
        db = SessionLocal()
        try:
            # Güvenli alım fonksiyonu
            def g(key, default=None):
                return patient_data.get(key, default)

            prediction_record = PredictionRecord(
                model_id=model_id,
                # snapshot kolonlar
                patient_external_id=g("id"),
                age=g("age"),
                gender=g("gender"),
                height=g("height"),
                weight=g("weight"),
                ap_hi=g("ap_hi"),
                ap_lo=g("ap_lo"),
                cholesterol=g("cholesterol"),
                gluc=g("gluc"),
                smoke=bool(g("smoke")),
                alco=bool(g("alco")),
                active=bool(g("active")),

                # mevcut alanlar
                patient_data=json.dumps(patient_data),
                prediction=int(prediction_result['prediction']),
                probability=float(prediction_result['probability']),
                risk_score=float(risk_score)
            )
            
            db.add(prediction_record)
            db.commit()
            db.refresh(prediction_record)
            
            # Appointment tablosuna da kayıt ekle
            self._create_appointment_record(db, prediction_record, patient_data)
            
            return prediction_record.id
            
        except Exception as e:
            db.rollback()
            raise Exception(f"Tahmin kayıt hatası: {str(e)}")
        finally:
            db.close()
    
    def _create_appointment_record(self, db, prediction_record: PredictionRecord, patient_data: Dict):
        """PredictionRecord'dan Appointment kaydı oluştur."""
        try:
            # patient_external_id'yi kullanarak User'ı bul veya oluştur
            patient_external_id = prediction_record.patient_external_id
            if not patient_external_id:
                return  # patient_external_id yoksa appointment oluşturma
            
            # Eğer patient_external_id zaten küçük bir sayı ise (mevcut user ID'si), direkt kullan
            # Eğer büyük bir sayı ise hash'le
            if patient_external_id < 1000000:  # Küçük sayı ise direkt kullan
                patient_id = patient_external_id
            else:  # Büyük sayı ise hash'le
                import hashlib
                patient_id = abs(hash(str(patient_external_id))) % 1000000  # 1 milyon altında ID
            
            # User'ı bul veya oluştur
            user = db.query(User).filter(User.id == patient_id).first()
            if not user:
                # User yoksa oluştur
                user = User(
                    id=patient_id,
                    username=f"patient_{patient_id}",
                    email=f"patient_{patient_id}@example.com",
                    password_hash="dummy_hash_for_patient",
                    full_name=f"Hasta {patient_id}",
                    role="patient",
                    is_active=True
                )
                db.add(user)
                db.flush()
            
            # Appointment oluştur
            appointment = Appointment(
                patient_id=user.id,
                prediction_record_id=prediction_record.id,
                age=prediction_record.age,
                gender=prediction_record.gender,
                height=prediction_record.height,
                weight=prediction_record.weight,
                ap_hi=prediction_record.ap_hi,
                ap_lo=prediction_record.ap_lo,
                cholesterol=prediction_record.cholesterol,
                gluc=prediction_record.gluc,
                smoke=int(prediction_record.smoke),
                alco=int(prediction_record.alco),
                active=int(prediction_record.active),
                risk_score=prediction_record.risk_score,
                priority_score=prediction_record.risk_score,
                prediction_label=prediction_record.prediction,
                prediction_probability=prediction_record.probability,
                model_name=f"Model_{prediction_record.model_id}",
                queue_position=1,  # Varsayılan sıra pozisyonu
                status="completed",  # UI'den gelen randevular tamamlanmış sayılır
                priority=self._determine_priority(prediction_record.risk_score),
                assigned_at=datetime.utcnow()
            )
            
            db.add(appointment)
            db.commit()
            
        except Exception as e:
            print(f"Appointment oluşturma hatası: {e}")
            # Hata durumunda rollback yapma, sadece logla
    
    def _determine_priority(self, risk_score: float) -> str:
        """Risk skoruna göre öncelik belirle"""
        if risk_score >= 0.8:
            return "urgent"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "normal"
        else:
            return "low"
    
    async def get_best_model_from_database(self) -> Optional[Dict[str, Any]]:
        """Veritabanından en iyi performanslı modeli getir."""
        db = SessionLocal()
        try:
            # En yüksek F1 skoruna sahip aktif modeli bul
            best_model = db.query(ModelRecord).filter(
                ModelRecord.is_active == True
            ).order_by(ModelRecord.f1_score.desc()).first()
            
            if not best_model:
                return None
            
            return {
                'id': best_model.id,
                'model_name': best_model.model_name,
                'model_type': best_model.model_type,
                'accuracy': float(best_model.accuracy) if best_model.accuracy else 0.0,
                'precision': float(best_model.precision) if best_model.precision else 0.0,
                'recall': float(best_model.recall) if best_model.recall else 0.0,
                'f1_score': float(best_model.f1_score) if best_model.f1_score else 0.0,
                'roc_auc': float(best_model.roc_auc) if best_model.roc_auc else None,
                'model_path': best_model.model_path,
                'feature_names': json.loads(best_model.feature_names) if best_model.feature_names else [],
                'training_params': json.loads(best_model.training_params) if best_model.training_params else {},
                'created_at': best_model.created_at.isoformat() if best_model.created_at else None,
                'is_active': best_model.is_active
            }
            
        except Exception as e:
            print(f"En iyi model getirme hatasi: {e}")
            return None
        finally:
            db.close()

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Kullanılabilir modelleri listele."""
        db = SessionLocal()
        try:
            models = db.query(ModelRecord).filter(ModelRecord.is_active == True).all()
            
            model_list = []
            for model in models:
                model_list.append({
                    'id': int(model.id) if model.id is not None else None,
                    'model_name': model.model_name,
                    'model_type': model.model_type,
                    'accuracy': float(model.accuracy) if model.accuracy is not None else None,
                    'f1_score': float(model.f1_score) if model.f1_score is not None else None,
                    'created_at': model.created_at.isoformat(),
                    'feature_count': int(len(model.feature_names)) if model.feature_names else 0
                })
            
            return self._convert_numpy_types(model_list)
            
        except Exception as e:
            print(f"Model listesi hatası: {e}")
            return []
        finally:
            db.close()

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

# Global service instance
prediction_service = PredictionService()
