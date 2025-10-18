# Kalp Krizi Risk Tahmin Modeli

Bu proje, kalp krizi riskini tahmin etmek için makine öğrenmesi modeli geliştirmeyi amaçlamaktadır. **Outlier'lı ve outliersız veriler için karşılaştırmalı analiz** yaparak en optimal modeli belirler. **GPU desteği** ile hızlandırılmış eğitim imkanı sunar.

## Sistem Mimarisi ve Teknik Altyapı

### Genel Mimari Yapı
Bu proje **mikroservis benzeri modüler mimari** ile tasarlanmıştır:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                             │
├─────────────────────────────────────────────────────────────┤
│  React Frontend (Admin Panel + Patient Panel)              │
│  • Material-UI Components                                  │
│  • TypeScript + Bootstrap Styling                          │
│  • Responsive Design                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   API GATEWAY LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI Backend                                           │
│  • RESTful API Endpoints                                   │
│  • Authentication & Authorization                          │
│  • Request/Response Validation (Pydantic)                  │
│  • Error Handling & Logging                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 BUSINESS LOGIC LAYER                       │
├─────────────────────────────────────────────────────────────┤
│  Service Layer                                             │
│  • Prediction Service (Risk Analysis)                      │
│  • Model Training Service (ML Pipeline)                    │
│  • Model Comparison Service (Performance Analysis)         │
│  • Data Processing Service (Feature Engineering)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA ACCESS LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  • SQLAlchemy ORM                                          │
│  • SQLite Database (Models & Predictions)                  │
│  • File System (Model Storage)                             │
│  • Data Pipeline (Advanced Feature Engineering)            │
└─────────────────────────────────────────────────────────────┘
```

### Detaylı Bileşen Analizi

#### 1. Frontend Katmanı (React + TypeScript)
```
frontend/src/
├── components/
│   ├── admin/              # Admin panel bileşenleri
│   │   ├── ModelComparison.tsx
│   │   ├── ModelTraining.tsx
│   │   └── Dashboard.tsx
│   ├── patient/            # Hasta paneli bileşenleri
│   │   ├── PredictionForm.tsx
│   │   ├── PredictionResult.tsx
│   │   └── RiskVisualization.tsx
│   └── shared/             # Paylaşılan bileşenler
│       ├── Layout.tsx
│       ├── LoadingSpinner.tsx
│       └── ErrorBoundary.tsx
├── services/
│   └── api.ts              # HTTP client (Axios)
├── types/
│   └── api.ts              # TypeScript tip tanımları
└── styles/
    └── bootstrap-custom.css # Bootstrap-inspired styling
```

**Özellikler:**
- **Responsive Design**: Mobile-first yaklaşım
- **Type Safety**: Full TypeScript desteği
- **State Management**: React Hooks + Context API
- **Error Handling**: Centralized error boundary
- **Loading States**: Skeleton loaders ve progress indicators

#### 2. API Katmanı (FastAPI)
```
api/
├── main.py                 # FastAPI uygulama entry point
├── routes/
│   ├── prediction.py       # POST /api/v1/prediction/predict
│   ├── model_training.py   # POST /api/v1/training/train
│   ├── model_comparison.py # GET /api/v1/models/compare
│   └── health.py          # GET /health (health check)
├── services/
│   ├── prediction_service.py      # Risk analizi iş mantığı
│   ├── model_training_service.py  # ML model eğitimi
│   ├── model_comparison_service.py # Model karşılaştırma
│   └── data_processing_service.py # Veri işleme
├── core/
│   ├── config.py          # Konfigürasyon yönetimi
│   ├── database.py        # SQLAlchemy modelleri
│   └── security.py        # Authentication/Authorization
└── middleware/
    ├── cors.py            # CORS ayarları
    ├── logging.py         # Request/Response logging
    └── error_handler.py   # Global error handling
```

**API Endpoints:**
```python
# Prediction Endpoints
POST /api/v1/prediction/predict
GET  /api/v1/prediction/history/{patient_id}

# Model Management
POST /api/v1/training/train/{model_type}
POST /api/v1/training/train/auto
GET  /api/v1/models/list
GET  /api/v1/models/compare

# Health & Monitoring
GET  /health
GET  /metrics
GET  /api/v1/system/status
```

#### 3. ML/Veri Katmanı
```
src/
├── data/
│   ├── advanced_feature_engineering.py  # Gelişmiş özellik üretimi
│   ├── preprocessor.py                  # Veri ön işleme
│   └── feature_engineering.py          # Temel feature engineering
├── analysis/
│   ├── hyperparameter_tuning.py        # Model optimizasyonu
│   ├── ensemble_methods.py             # Ensemble teknikleri
│   └── data_analysis.py                # Model analizi
├── models/
│   ├── base_model.py                   # Temel model sınıfı
│   ├── ensemble_model.py               # Ensemble model implementasyonu
│   └── calibration.py                  # Model kalibrasyonu
└── utils/
    ├── data_loader.py                  # Veri yükleme yardımcıları
    ├── metrics.py                      # Özel metrikler
    └── visualization.py                # Grafik üretimi
```

#### 4. Veri Depolama Katmanı
```sql
-- Model Records Table
CREATE TABLE model_records (
    id INTEGER PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    roc_auc REAL NOT NULL,
    brier_score REAL,
    log_loss REAL,
    ece_score REAL,
    feature_names TEXT,  -- JSON format
    training_params TEXT, -- JSON format
    model_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Prediction Records Table
CREATE TABLE prediction_records (
    id INTEGER PRIMARY KEY,
    patient_id VARCHAR(100),
    model_id INTEGER,
    input_data TEXT,  -- JSON format
    prediction REAL,
    risk_score REAL,
    risk_level VARCHAR(20),
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES model_records (id)
);
```

### Performans ve Ölçeklenebilirlik

#### 1. Asenkron İşlemler
```python
# Background task processing
from concurrent.futures import ThreadPoolExecutor
import asyncio

class AsyncModelTraining:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def train_model_async(self, model_type: str, data_path: str):
        """Asenkron model eğitimi"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._train_model_sync, 
            model_type, 
            data_path
        )
```

#### 2. Caching Stratejisi
```python
# Redis cache (opsiyonel)
from functools import lru_cache
import pickle

@lru_cache(maxsize=128)
def load_model_cached(model_path: str):
    """Model cache'leme"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# API response caching
@app.middleware("http")
async def cache_middleware(request: Request, call_next):
    if request.method == "GET" and "models/compare" in str(request.url):
        # Cache model comparison results
        pass
    response = await call_next(request)
    return response
```

#### 3. Veri Pipeline Optimizasyonu
```python
class OptimizedDataPipeline:
    def __init__(self):
        self.preprocessor = StandardScaler()
        self.feature_selector = SelectKBest(k=20)
        self.cached_features = {}
    
    def process_data_chunk(self, chunk_size: int = 10000):
        """Chunk-based data processing"""
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            processed_chunk = self._process_chunk(chunk)
            yield processed_chunk
    
    def parallel_feature_engineering(self, data: pd.DataFrame):
        """Paralel feature engineering"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(self._engineer_age_features, data),
                executor.submit(self._engineer_bmi_features, data),
                executor.submit(self._engineer_bp_features, data),
                executor.submit(self._engineer_lifestyle_features, data)
            ]
            results = [future.result() for future in futures]
        return pd.concat(results, axis=1)
```

### Güvenlik ve Monitoring

#### 1. API Güvenliği
```python
# Rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/prediction/predict")
@limiter.limit("10/minute")  # Rate limiting
async def predict_risk(request: Request, patient_data: PatientData):
    # Input validation
    validated_data = validate_patient_data(patient_data)
    # Sanitize inputs
    sanitized_data = sanitize_inputs(validated_data)
    # Process prediction
    result = await prediction_service.predict(sanitized_data)
    return result
```

#### 2. Monitoring ve Logging
```python
import structlog
from prometheus_client import Counter, Histogram

# Metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total predictions')
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Prediction duration')

# Structured logging
logger = structlog.get_logger()

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(
        "request_processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response
```

#### 3. Health Checks
```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": await check_database_health(),
            "models": await check_models_health(),
            "disk_space": await check_disk_space(),
            "memory": await check_memory_usage()
        }
    }
    return health_status
```

## Teknik Yeterlilik ve Algoritma Detayları

### Makine Öğrenmesi Algoritmaları
Bu proje **9 farklı ML algoritması** ile kapsamlı model karşılaştırması yapmaktadır:

#### 1. Ensemble Methods (Toplu Öğrenme)
- **Random Forest**: Bootstrap aggregating ile 100+ ağaç, feature importance analizi
- **Gradient Boosting**: Adaptive boosting ile sıralı öğrenme, overfitting koruması
- **XGBoost**: Extreme gradient boosting, GPU desteği ile hızlandırılmış eğitim
- **LightGBM**: Leaf-wise tree growth, hafıza verimli gradient boosting
- **CatBoost**: Kategorik değişken optimizasyonu, overfitting önleme
- **AdaBoost**: Adaptive boosting, zayıf öğrenicilerin birleştirilmesi

#### 2. Klasik Algoritmalar
- **Logistic Regression**: L1/L2 regularization, probabilistic output
- **Support Vector Machine**: RBF kernel, probability calibration
- **Ensemble Methods**: Voting, bagging ve stacking kombinasyonları

### Risk Skoru Algoritması ve Sıralama Sistemi

#### Risk Skoru Hesaplama Algoritması
```python
def calculate_risk_score(patient_data, model_probability):
    """
    Kapsamlı risk skoru hesaplama algoritması
    """
    # 1. Temel model olasılığı (0-1 arası)
    base_risk = model_probability
    
    # 2. Yaş risk faktörü (exponential scaling)
    age_factor = min(2.0, (patient_data['age_years'] / 50) ** 1.5)
    
    # 3. BMI risk faktörü (U-shaped curve)
    bmi = patient_data['weight'] / (patient_data['height']/100) ** 2
    bmi_risk = 1.0 + abs(bmi - 22.5) * 0.02  # Optimal BMI: 22.5
    
    # 4. Tansiyon risk faktörü
    bp_risk = 1.0
    if patient_data['ap_hi'] > 140 or patient_data['ap_lo'] > 90:
        bp_risk = 1.5 + (patient_data['ap_hi'] - 140) * 0.01
    
    # 5. Yaşam tarzı risk faktörleri
    lifestyle_risk = 1.0
    if patient_data['smoke']: lifestyle_risk += 0.3
    if patient_data['alco']: lifestyle_risk += 0.2
    if not patient_data['active']: lifestyle_risk += 0.2
    
    # 6. Kolesterol ve glikoz risk faktörleri
    metabolic_risk = 1.0
    if patient_data['cholesterol'] > 2: metabolic_risk += 0.2
    if patient_data['gluc'] > 2: metabolic_risk += 0.3
    
    # 7. Final risk skoru hesaplama (0-100 arası)
    final_risk = (base_risk * age_factor * bmi_risk * bp_risk * 
                 lifestyle_risk * metabolic_risk) * 100
    
    return min(100, max(0, final_risk))
```

#### Risk Seviyesi Sınıflandırması
```python
def classify_risk_level(risk_score):
    """
    Risk skoruna göre seviye belirleme
    """
    if risk_score < 20:
        return "DÜŞÜK RİSK", "#28a745"  # Yeşil
    elif risk_score < 40:
        return "ORTA RİSK", "#ffc107"   # Sarı
    elif risk_score < 60:
        return "YÜKSEK RİSK", "#fd7e14" # Turuncu
    else:
        return "ÇOK YÜKSEK RİSK", "#dc3545"  # Kırmızı
```

#### Sıralama ve Önceliklendirme Algoritması
```python
def prioritize_patients(patients_list):
    """
    Hastaları risk skoruna göre sıralama algoritması
    """
    # 1. Risk skoruna göre birincil sıralama (azalan)
    patients_sorted = sorted(patients_list, key=lambda x: x['risk_score'], reverse=True)
    
    # 2. Aynı risk skorunda yaş faktörü (artan - yaşlılar öncelikli)
    patients_sorted = sorted(patients_sorted, 
                           key=lambda x: (x['risk_score'], -x['age_years']), 
                           reverse=True)
    
    # 3. Kritik faktörler kontrolü
    for i, patient in enumerate(patients_sorted):
        priority_boost = 0
        if patient['ap_hi'] > 180: priority_boost += 10  # Hipertansif kriz
        if patient['smoke'] and patient['age_years'] > 50: priority_boost += 5
        if patient['cholesterol'] == 3: priority_boost += 3
        
        patient['final_priority'] = i + 1 + priority_boost
    
    # 4. Final sıralama
    return sorted(patients_sorted, key=lambda x: x['final_priority'])
```

### Model Değerlendirme Metrikleri

#### 1. Klasik Metrikler
- **Accuracy**: Genel doğruluk oranı
- **Precision**: Pozitif tahminlerin doğruluk oranı
- **Recall**: Gerçek pozitiflerin yakalanma oranı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması
- **ROC AUC**: Receiver Operating Characteristic alanı

#### 2. Kalibrasyon Metrikleri (İleri Seviye)
- **Brier Score**: Olasılık tahminlerinin kalitesi (0-1, düşük=iyi)
- **Log Loss**: Olasılık tabanlı hata metriği
- **ECE (Expected Calibration Error)**: Kalibrasyon hatası ölçümü
- **Calibration Curves**: Tahmin vs gerçek olasılık karşılaştırması

#### 3. Model Seçim Kriterleri
```python
def select_best_model(model_results):
    """
    En iyi model seçim algoritması
    """
    # 1. Kalibrasyon metrikleri ağırlığı
    calibration_weight = 0.4
    performance_weight = 0.6
    
    # 2. Brier Score ve Log Loss normalizasyonu
    best_brier = min(results['brier_score'] for results in model_results.values())
    best_logloss = min(results['log_loss'] for results in model_results.values())
    
    # 3. Composite score hesaplama
    for model_name, results in model_results.items():
        brier_score = results['brier_score'] / best_brier
        logloss_score = results['log_loss'] / best_logloss
        f1_score = results['f1_score']
        
        composite_score = (performance_weight * f1_score + 
                          calibration_weight * (2 - brier_score - logloss_score) / 2)
        
        results['composite_score'] = composite_score
    
    # 4. En iyi model seçimi
    best_model = max(model_results.items(), key=lambda x: x[1]['composite_score'])
    return best_model[0], best_model[1]
```

### Gelişmiş Feature Engineering

#### 1. Domain-Specific Features
- **Yaş Grupları**: Demografik segmentasyon
- **BMI Kategorileri**: Sağlık riski sınıflandırması
- **Tansiyon Sınıflandırması**: Hipertansiyon evreleri
- **Metabolik Sendrom**: Çoklu risk faktörü kombinasyonu

#### 2. İnteraksiyon Features
- **Yaş × BMI**: Yaşlanma ve obezite etkileşimi
- **Tansiyon × Yaş**: Kardiyovasküler risk yaş faktörü
- **Yaşam Tarzı × Yaş**: Risk faktörlerinin yaşla değişimi

#### 3. Polinomik Features
- **2. Derece Polinomlar**: Non-linear ilişkilerin yakalanması
- **Feature Scaling**: StandardScaler ile normalizasyon
- **Outlier Handling**: IQR ve domain-specific kurallar

### Sistem Performans Optimizasyonu

#### 1. GPU Hızlandırma
- **XGBoost GPU**: CUDA desteği ile 5-10x hızlanma
- **LightGBM GPU**: Hafıza verimli GPU kullanımı
- **Batch Processing**: Büyük veri setleri için optimizasyon

#### 2. Model Optimizasyonu
- **Hyperparameter Tuning**: Grid Search ve Random Search
- **Cross-Validation**: 5-fold CV ile robust değerlendirme
- **Early Stopping**: Overfitting önleme
- **Feature Selection**: Recursive feature elimination

#### 3. Veri Pipeline Optimizasyonu
- **Caching**: Sık kullanılan hesaplamaların önbellekleme
- **Parallel Processing**: Multi-core CPU kullanımı
- **Memory Management**: Büyük veri setleri için chunk processing
- **Incremental Learning**: Yeni verilerle model güncelleme

## Proje Yönetimi

### Mimari Tasarım ve Organizasyon
Bu proje **3-katmanlı mimari** ile tasarlanmıştır:

#### 1. Sunum Katmanı (Presentation Layer)
- **Frontend**: React 18 + TypeScript + Material-UI
- **Admin Panel**: Model yönetimi, karşılaştırma ve otomatik eğitim
- **Hasta Panel**: Risk analizi formu ve sonuç görüntüleme
- **Responsive Design**: Bootstrap-inspired modern UI/UX

#### 2. İş Mantığı Katmanı (Business Logic Layer)
- **FastAPI Backend**: RESTful API mimarisi
- **Servis Katmanı**: İş kuralları ve ML entegrasyonu
- **Model Yönetimi**: Otomatik model seçimi ve kalibrasyon
- **Veri Validasyonu**: Pydantic ile tip güvenliği

#### 3. Veri Katmanı (Data Layer)
- **SQLite Veritabanı**: Model kayıtları ve tahmin geçmişi
- **Model Depolama**: Pickle formatında serileştirilmiş modeller
- **Veri Pipeline**: Gelişmiş feature engineering ve ön işleme

### Proje Yaşam Döngüsü Yönetimi

#### Geliştirme Aşaması
- **Sürüm Kontrolü**: Git ile versiyonlama
- **Kod Kalitesi**: Type hints, docstrings ve linting
- **Test Stratejisi**: Unit testler ve entegrasyon testleri
- **Dokümantasyon**: Kapsamlı README ve API dokümantasyonu

#### Deployment ve DevOps
- **Geliştirme Ortamı**: `uvicorn --reload` ile hot-reload
- **Üretim Ortamı**: Gunicorn + NGINX reverse proxy önerilir
- **Konteynerizasyon**: Docker desteği (opsiyonel)
- **Monitoring**: Log yönetimi ve performans izleme

#### Kalite Güvencesi
- **Veri Doğrulama**: Domain-specific kurallar ve outlier kontrolü
- **Model Validasyonu**: Cross-validation ve metrik bazlı değerlendirme
- **Hata Yönetimi**: Graceful error handling ve kullanıcı dostu mesajlar
- **Güvenlik**: Input sanitization ve SQL injection koruması

### Risk Yönetimi ve Güvenilirlik

#### Teknik Riskler
- **Model Overfitting**: Cross-validation ve regularization teknikleri
- **Veri Kalitesi**: Kapsamlı veri doğrulama ve temizleme pipeline'ı
- **Performans**: GPU desteği ve model optimizasyonu
- **Skalabilite**: Asenkron işlemler ve veritabanı optimizasyonu

#### İş Süreçleri
- **Sürekli Entegrasyon**: Otomatik test ve deployment
- **Backup Stratejisi**: Model ve veri yedekleme
- **Disaster Recovery**: Sistem kurtarma planları
- **Dokümantasyon**: Sürekli güncellenen teknik dokümantasyon


## 🚀 GPU Desteği

Bu proje artık **GPU desteği** ile gelmektedir! XGBoost ve LightGBM modelleri GPU üzerinde çalıştırılarak eğitim süreleri önemli ölçüde azaltılabilir.

### GPU Gereksinimleri
- NVIDIA GPU (CUDA uyumlu)
- CUDA Toolkit 11.0+
- GPU destekli Python paketleri

### GPU Kurulumu
Detaylı kurulum rehberi için [GPU_SETUP.md](GPU_SETUP.md) dosyasını inceleyin.

### GPU Testi
GPU desteğini test etmek için:
```bash
cd src
python test_gpu.py
```

## Proje Yapısı

```
cardiyovask/
├── data/                   # Veri dosyaları
│   └── cardiokaggle.csv   # Ham veri
├── src/                   # Kaynak kodlar
│   ├── data/             # Veri işleme modülleri
│   │   ├── cardiokaggle.csv
│   │   ├── feature_engineering.py
│   │   └── preprocessor.py
│   ├── analysis/         # Veri analizi modülleri
│   │   └── data_analysis.py
│   ├── utils/            # Yardımcı fonksiyonlar
│   │   └── data_loader.py
│   ├── main.py           # Ana çalıştırma dosyası
│   ├── clean_data.py     # Veri temizleme scripti
│   ├── clean_bp_outliers.py
│   ├── test_analysis.py
│   └── test_cleaned_data.py
├── requirements.txt      # Python bağımlılıkları
├── GPU_SETUP.md         # GPU kurulum rehberi
└── README.md            # Proje dokümantasyonu
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Karşılaştırmalı Analiz Pipeline

Outlier'lı ve outliersız veriler için karşılaştırmalı analiz çalıştırmak için:

```bash
cd src
python main.py
```

### GPU Destekli Optimizasyon Pipeline

GPU desteği ile hızlandırılmış optimizasyon için:

```bash
cd src
python main_optimized.py
```

### Ayrı Modüller

**Veri Ön İşleme için:**
```bash
python src/data/preprocessor.py
```

**Karşılaştırmalı Analiz için:**
```bash
python src/analysis/data_analysis.py
```

**Veri Temizleme için:**
```bash
python src/clean_data.py
```

**Test Analizi için:**
```bash
python src/test_analysis.py
```

## Özellikler

### Veri İşleme (preprocessor.py)
- ✅ **Yaş dönüşümü**: Gün formatından yıl formatına çevirme (age → age_years)
- ✅ **Duplike kayıt kontrolü**: Tekrarlanan kayıtları tespit etme ve raporlama
- ✅ **Eksik değer analizi**: Detaylı eksik değer analizi ve görselleştirme
- ✅ **Domain-specific outlier tespiti**: Alan bilgisi kurallarına göre outlier tespiti
- ✅ **Kategorik değişken doğrulama**: Geçerli değer aralıklarını kontrol etme
- ✅ **Sayısal değişken doğrulama**: Fiziksel sınırlar içinde değer kontrolü
- ✅ **Tansiyon kuralı kontrolü**: ap_hi ≥ ap_lo kuralını doğrulama
- ✅ **Outlier temizleme seçeneği**: Domain-specific kurallara göre temizleme
- ✅ **Veri kalitesi raporu**: Kapsamlı veri kalitesi analizi
- ✅ **Görselleştirme**: Eksik değer ve outlier analizleri için grafikler

### Modelleme
- ✅ Random Forest Classifier
- ✅ Gradient Boosting Classifier
- ✅ Logistic Regression
- ✅ Support Vector Machine (SVM)
- ✅ **XGBoost Classifier** (GPU desteği ile!)
- ✅ **LightGBM** (GPU desteği ile!)

### Değerlendirme Metrikleri
- ✅ Accuracy, Precision, Recall, F1-Score
- ✅ ROC AUC Score
- ✅ Confusion Matrix
- ✅ Feature Importance (Random Forest ve XGBoost)

### Karşılaştırmalı Analiz
- ✅ **Outlier'lı vs Outliersız veri karşılaştırması**
- ✅ Her iki veri seti için ayrı model eğitimi
- ✅ Performans metriklerinin karşılaştırmalı analizi
- ✅ En iyi model seçimi (her veri seti için)
- ✅ Outlier temizlemenin etkisinin ölçülmesi

### Görselleştirme
- ✅ Confusion Matrix grafikleri (her veri seti için)
- ✅ ROC eğrileri (her veri seti için)
- ✅ Model performans karşılaştırması (her veri seti için)
- ✅ Feature importance grafiği (Random Forest ve XGBoost)
- ✅ **Karşılaştırmalı analiz grafikleri** (Yeni!)

## Çıktılar

Pipeline çalıştırıldığında aşağıdaki çıktılar oluşturulur:

### 1. Konsol Çıktıları
- **Veri ön işleme adımları** (preprocessor.py)
  - Duplike kayıt sayısı ve oranı
  - Eksik değer analizi (kolon bazında)
  - Domain-specific outlier tespiti
  - Veri kalitesi raporu
- **Model eğitimi ve değerlendirme** (data_analysis.py)
  - Model performans metrikleri (her iki veri seti için)
  - X_train.shape, X_test.shape (her iki veri seti için)
  - **Karşılaştırmalı sonuçlar tablosu**
  - **En iyi model karşılaştırması**
  - **Outlier temizlemenin performans etkisi**

### 2. Grafik Dosyaları
- **Veri ön işleme grafikleri** (preprocessor.py)
  - `missing_values_analysis.png` - Eksik değer analizi
  - `domain_outliers_analysis.png` - Sayısal değişken outlier'ları
  - `categorical_outliers_analysis.png` - Kategorik değişken outlier'ları
- **Model analizi grafikleri** (data_analysis.py)
  - `confusion_matrices.png` - Tüm modellerin confusion matrix'leri
  - `roc_curves.png` - ROC eğrileri
  - `metrics_comparison.png` - Model performans karşılaştırması
  - `rf_feature_importance.png` - Random Forest feature importance
  - `xgb_feature_importance.png` - XGBoost feature importance
  - `comparative_analysis.png` - Karşılaştırmalı analiz grafikleri

## Veri Seti

Kullanılan veri seti aşağıdaki kolonları içerir:
- `id`: Hasta ID
- `age`: Yaş (gün cinsinden)
- `gender`: Cinsiyet
- `height`: Boy
- `weight`: Kilo
- `ap_hi`: Sistolik tansiyon
- `ap_lo`: Diastolik tansiyon
- `cholesterol`: Kolesterol seviyesi
- `gluc`: Glikoz seviyesi
- `smoke`: Sigara kullanımı
- `alco`: Alkol kullanımı
- `active`: Fiziksel aktivite
- `cardio`: Kardiyovasküler hastalık (hedef değişken)

## Domain-Specific Outlier Kuralları

### Sayısal Değişkenler
- **age (gün)**: 6570 ≤ age ≤ 36500 (≈ 18–100 yaş), > 43800 (120 yaş) kesin hatalı
- **height (cm)**: 120 ≤ height ≤ 220
- **weight (kg)**: 30 ≤ weight ≤ 200, > 300 kesin hatalı
- **ap_hi (sistolik)**: 80 ≤ ap_hi ≤ 240
- **ap_lo (diyastolik)**: 40 ≤ ap_lo ≤ 150
- **Tansiyon kuralı**: ap_hi ≥ ap_lo (sistolik ≥ diyastolik)

### Kategorik Değişkenler
- **gender**: {1, 2}
- **cholesterol**: {1, 2, 3}
- **gluc**: {1, 2, 3}
- **smoke, alco, active, cardio**: {0, 1}

## Karşılaştırmalı Analiz Özellikleri

### Outlier Temizleme
- **Domain-Specific Metod**: Alan bilgisi kurallarına göre outlier tespiti ve temizleme
- **IQR Metodu**: Q1 - 1.5*IQR ve Q3 + 1.5*IQR aralığı dışındaki değerler (eski metod)
- **Temizlenen Kolonlar**: Tüm sayısal ve kategorik değişkenler
- **Karşılaştırma**: Outlier'lı ve outliersız veriler için ayrı model eğitimi

### Model Karşılaştırması
- Her veri seti için 5 farklı model eğitilir
- F1-Score'a göre en iyi model seçilir
- Outlier temizlemenin performans etkisi ölçülür

### Görselleştirme
- Karşılaştırmalı bar grafikleri
- Her metrik için ayrı karşılaştırma
- Outlier'lı vs outliersız performans farkı

## Gereksinimler

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- **xgboost >= 1.6.0** (Yeni!) 
#############################################################
================================================================================
KALP KRİZİ RİSK TAHMİN MODELİ - KARŞILAŞTIRMALI ANALİZ PIPELINE
================================================================================
Veri dosyası: /Users/aybukealtuntas/Desktop/cardiovasktrain/cardiyovask/src/data/cardiokaggle.csv

==================================================
1. VERİ İŞLEME VE FEATURE ENGINEERING
==================================================

--- Outlier'lı Verilerle İşleme ---
=== KALP KRİZİ RİSK TAHMİN MODELİ - OUTLIER'LAR İLE VERİ İŞLEME ===

Veri başarıyla yüklendi! Boyut: (70000, 13)
Kolonlar: ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
Yaş dönüşümü tamamlandı. Örnek değerler:
Gün: [18393, 20228, 18857] -> Yıl: [50, 55, 51]

=== EKSİK DEĞER ANALİZİ ===
Toplam eksik değer: 0
Eksik değer bulunmamaktadır.
Eksik değerler silindi. Yeni boyut: (70000, 14)

=== OUTLIER ANALİZİ ===
height: 519 outlier (0.74%)
weight: 1819 outlier (2.60%)
ap_hi: 1435 outlier (2.05%)
ap_lo: 4632 outlier (6.62%)
cholesterol: 0 outlier (0.00%)
gluc: 10521 outlier (15.03%)

Toplam outlier sayısı: 18926

=== KATEGORİK DEĞİŞKEN ENCODING ===
gender: 2 benzersiz değer -> [1, 2] -> [0, 1]
smoke: 2 benzersiz değer -> [0, 1] -> [0, 1]
alco: 2 benzersiz değer -> [0, 1] -> [0, 1]
active: 2 benzersiz değer -> [0, 1] -> [0, 1]

=== SÜREKLİ DEĞİŞKEN ÖLÇEKLEME ===
Ölçeklenen kolonlar: ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
StandardScaler uygulandı (ortalama=0, standart sapma=1)

Feature matrix boyutu: (70000, 11)
Hedef değişken boyutu: (70000,)
Feature kolonları: ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years']

=== VERİ AYRIMI ===
Eğitim seti: (56000, 11)
Test seti: (14000, 11)
Eğitim hedef: (56000,)
Test hedef: (14000,)

=== OUTLIER'LAR İLE VERİ İŞLEME TAMAMLANDI ===

--- Outlier'lar Çıkarılarak İşleme ---
=== KALP KRİZİ RİSK TAHMİN MODELİ - OUTLIER'LAR ÇIKARILARAK VERİ İŞLEME ===

Veri başarıyla yüklendi! Boyut: (70000, 13)
Kolonlar: ['id', 'age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
Yaş dönüşümü tamamlandı. Örnek değerler:
Gün: [18393, 20228, 18857] -> Yıl: [50, 55, 51]

=== EKSİK DEĞER ANALİZİ ===
Toplam eksik değer: 0
Eksik değer bulunmamaktadır.
Eksik değerler silindi. Yeni boyut: (70000, 14)

=== OUTLIER ANALİZİ ===
height: 519 outlier (0.74%)
weight: 1819 outlier (2.60%)
ap_hi: 1435 outlier (2.05%)
ap_lo: 4632 outlier (6.62%)
cholesterol: 0 outlier (0.00%)
gluc: 10521 outlier (15.03%)

Toplam outlier sayısı: 18926

=== OUTLIER TEMİZLEME ===
height: 519 outlier çıkarıldı
weight: 1819 outlier çıkarıldı
ap_hi: 1435 outlier çıkarıldı
ap_lo: 4632 outlier çıkarıldı
cholesterol: 0 outlier çıkarıldı
gluc: 10521 outlier çıkarıldı
Toplam 18926 outlier çıkarıldı
Orijinal boyut: (70000, 14) -> Temizlenmiş boyut: (53408, 14)

=== KATEGORİK DEĞİŞKEN ENCODING ===
gender: 2 benzersiz değer -> [1, 2] -> [0, 1]
smoke: 2 benzersiz değer -> [0, 1] -> [0, 1]
alco: 2 benzersiz değer -> [0, 1] -> [0, 1]
active: 2 benzersiz değer -> [0, 1] -> [0, 1]

=== SÜREKLİ DEĞİŞKEN ÖLÇEKLEME ===
Ölçeklenen kolonlar: ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc']
StandardScaler uygulandı (ortalama=0, standart sapma=1)

Feature matrix boyutu: (53408, 11)
Hedef değişken boyutu: (53408,)
Feature kolonları: ['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'age_years']

=== VERİ AYRIMI ===
Eğitim seti: (42726, 11)
Test seti: (10682, 11)
Eğitim hedef: (42726,)
Test hedef: (10682,)

=== OUTLIER'LAR ÇIKARILARAK VERİ İŞLEME TAMAMLANDI ===

==================================================
2. KARŞILAŞTIRMALI MODEL EĞİTİMİ VE DEĞERLENDİRME
==================================================
================================================================================
KALP KRİZİ RİSK TAHMİN MODELİ - KARŞILAŞTIRMALI ANALİZ
================================================================================

========================================
1. OUTLIER'LAR İLE ANALİZ
========================================
=== KALP KRİZİ RİSK TAHMİN MODELİ - TAM ANALİZ (WITH_OUTLIERS) ===

=== MODEL EĞİTİMİ ===

Random Forest eğitiliyor...
Random Forest eğitimi tamamlandı.

Gradient Boosting eğitiliyor...
Gradient Boosting eğitimi tamamlandı.

Logistic Regression eğitiliyor...
Logistic Regression eğitimi tamamlandı.

SVM eğitiliyor...
SVM eğitimi tamamlandı.

XGBoost eğitiliyor...
XGBoost eğitimi tamamlandı.

Tüm modeller eğitildi!

=== MODEL DEĞERLENDİRMESİ ===

Random Forest değerlendiriliyor...
Accuracy: 0.7077
Precision: 0.7093
Recall: 0.7033
F1-Score: 0.7063
ROC AUC: 0.7635

Gradient Boosting değerlendiriliyor...
Accuracy: 0.7336
Precision: 0.7507
Recall: 0.6990
F1-Score: 0.7239
ROC AUC: 0.7997

Logistic Regression değerlendiriliyor...
Accuracy: 0.7141
Precision: 0.7318
Recall: 0.6754
F1-Score: 0.7024
ROC AUC: 0.7783

SVM değerlendiriliyor...
Accuracy: 0.7294
Precision: 0.7440
Recall: 0.6990
F1-Score: 0.7208
ROC AUC: 0.7859

XGBoost değerlendiriliyor...
Accuracy: 0.7339
Precision: 0.7530
Recall: 0.6958
F1-Score: 0.7233
ROC AUC: 0.8000

=== EN İYİ MODEL SEÇİMİ ===
Random Forest: F1-Score = 0.7063
Gradient Boosting: F1-Score = 0.7239
Logistic Regression: F1-Score = 0.7024
SVM: F1-Score = 0.7208
XGBoost: F1-Score = 0.7233

En iyi model: Gradient Boosting (F1-Score: 0.7239)

=== Gradient Boosting - DETAYLI SINIFLANDIRMA RAPORU ===
                     precision    recall  f1-score   support

Kardiyovasküler Yok       0.72      0.77      0.74      7004
Kardiyovasküler Var       0.75      0.70      0.72      6996

           accuracy                           0.73     14000
          macro avg       0.73      0.73      0.73     14000
       weighted avg       0.73      0.73      0.73     14000


Görselleştirmeler oluşturuluyor...
2025-08-11 21:20:37.709 python[80049:13502621] +[IMKClient subclass]: chose IMKClient_Legacy
2025-08-11 21:20:37.709 python[80049:13502621] +[IMKInputSession subclass]: chose IMKInputSession_Legacy

=== ANALİZ TAMAMLANDI (WITH_OUTLIERS) ===
En iyi model: Gradient Boosting
En iyi F1-Score: 0.7239

========================================
2. OUTLIER'LAR ÇIKARILARAK ANALİZ
========================================
=== KALP KRİZİ RİSK TAHMİN MODELİ - TAM ANALİZ (WITHOUT_OUTLIERS) ===

=== MODEL EĞİTİMİ ===

Random Forest eğitiliyor...
Random Forest eğitimi tamamlandı.

Gradient Boosting eğitiliyor...
Gradient Boosting eğitimi tamamlandı.

Logistic Regression eğitiliyor...
Logistic Regression eğitimi tamamlandı.

SVM eğitiliyor...
SVM eğitimi tamamlandı.

XGBoost eğitiliyor...
XGBoost eğitimi tamamlandı.

Tüm modeller eğitildi!

=== MODEL DEĞERLENDİRMESİ ===

Random Forest değerlendiriliyor...
Accuracy: 0.7070
Precision: 0.7006
Recall: 0.6714
F1-Score: 0.6857
ROC AUC: 0.7634

Gradient Boosting değerlendiriliyor...
Accuracy: 0.7395
Precision: 0.7570
Recall: 0.6667
F1-Score: 0.7090
ROC AUC: 0.8056

Logistic Regression değerlendiriliyor...
Accuracy: 0.7344
Precision: 0.7653
Recall: 0.6376
F1-Score: 0.6956
ROC AUC: 0.7984

SVM değerlendiriliyor...
Accuracy: 0.7368
Precision: 0.7798
Recall: 0.6232
F1-Score: 0.6928
ROC AUC: 0.7870

XGBoost değerlendiriliyor...
Accuracy: 0.7416
Precision: 0.7658
Recall: 0.6586
F1-Score: 0.7082
ROC AUC: 0.8046

=== EN İYİ MODEL SEÇİMİ ===
Random Forest: F1-Score = 0.6857
Gradient Boosting: F1-Score = 0.7090
Logistic Regression: F1-Score = 0.6956
SVM: F1-Score = 0.6928
XGBoost: F1-Score = 0.7082

En iyi model: Gradient Boosting (F1-Score: 0.7090)

=== Gradient Boosting - DETAYLI SINIFLANDIRMA RAPORU ===
                     precision    recall  f1-score   support

Kardiyovasküler Yok       0.73      0.81      0.76      5597
Kardiyovasküler Var       0.76      0.67      0.71      5085

           accuracy                           0.74     10682
          macro avg       0.74      0.74      0.74     10682
       weighted avg       0.74      0.74      0.74     10682


Görselleştirmeler oluşturuluyor...
2025-08-11 21:52:41.536 python[80049:13502621] _TIPropertyValueIsValid called with 16 on nil context!
2025-08-11 21:52:41.536 python[80049:13502621] imkxpc_getApplicationProperty:reply: called with incorrect property value 16, bailing.
2025-08-11 21:52:41.536 python[80049:13502621] Text input context does not respond to _valueForTIProperty:
2025-08-11 21:54:29.329 python[80049:13502621] _TIPropertyValueIsValid called with 16 on nil context!
2025-08-11 21:54:29.329 python[80049:13502621] imkxpc_getApplicationProperty:reply: called with incorrect property value 16, bailing.
2025-08-11 21:54:29.329 python[80049:13502621] Text input context does not respond to _valueForTIProperty:

=== ANALİZ TAMAMLANDI (WITHOUT_OUTLIERS) ===
En iyi model: Gradient Boosting
En iyi F1-Score: 0.7090

========================================
3. KARŞILAŞTIRMALI SONUÇLAR
========================================

=== KARŞILAŞTIRMALI SONUÇLAR TABLOSU ===
              Model  Accuracy  Precision   Recall  F1-Score  ROC AUC        Data_Type
      Random Forest  0.707714   0.709343 0.703259  0.706288 0.763495    With Outliers
  Gradient Boosting  0.733571   0.750691 0.698971  0.723908 0.799737    With Outliers
Logistic Regression  0.714071   0.731764 0.675386  0.702446 0.778264    With Outliers
                SVM  0.729357   0.743953 0.698971  0.720761 0.785948    With Outliers
            XGBoost  0.733929   0.752978 0.695826  0.723275 0.799985    With Outliers
      Random Forest  0.706984   0.700595 0.671386  0.685680 0.763354 Without Outliers
  Gradient Boosting  0.739468   0.757034 0.666667  0.708983 0.805611 Without Outliers
Logistic Regression  0.734413   0.765345 0.637561  0.695634 0.798412 Without Outliers
                SVM  0.736847   0.779774 0.623206  0.692753 0.787034 Without Outliers
            XGBoost  0.741621   0.765836 0.658604  0.708184 0.804620 Without Outliers

=== EN İYİ MODEL KARŞILAŞTIRMASI ===
Outlier'lı veriler: Gradient Boosting (F1: 0.7239)
Outlier'lar çıkarılmış: Gradient Boosting (F1: 0.7090)
Outlier temizleme ile F1-Score azalışı: -0.0149

================================================================================
FİNAL KARŞILAŞTIRMALI SONUÇLAR
================================================================================

OUTLIER'LAR İLE:
X_train.shape: (56000, 11)
X_test.shape: (14000, 11)
En iyi model: Gradient Boosting
En iyi F1-Score: 0.7239

Outlier Analizi (çıkarılmadan):
  height: 519 outlier
  weight: 1819 outlier
  ap_hi: 1435 outlier
  ap_lo: 4632 outlier
  cholesterol: 0 outlier
  gluc: 10521 outlier

OUTLIER'LAR ÇIKARILARAK:
X_train.shape: (42726, 11)
X_test.shape: (10682, 11)
En iyi model: Gradient Boosting
En iyi F1-Score: 0.7090

Outlier Analizi (çıkarıldıktan sonra):
  height: 519 outlier (çıkarıldı)
  weight: 1819 outlier (çıkarıldı)
  ap_hi: 1435 outlier (çıkarıldı)
  ap_lo: 4632 outlier (çıkarıldı)
  cholesterol: 0 outlier (çıkarıldı)
  gluc: 10521 outlier (çıkarıldı)

Model Performans Karşılaştırması (Outlier'lı):
              Model  Accuracy  Precision   Recall  F1-Score  ROC AUC
      Random Forest  0.707714   0.709343 0.703259  0.706288 0.763495
  Gradient Boosting  0.733571   0.750691 0.698971  0.723908 0.799737
Logistic Regression  0.714071   0.731764 0.675386  0.702446 0.778264
                SVM  0.729357   0.743953 0.698971  0.720761 0.785948
            XGBoost  0.733929   0.752978 0.695826  0.723275 0.799985

Model Performans Karşılaştırması (Outlier'lar çıkarılmış):
              Model  Accuracy  Precision   Recall  F1-Score  ROC AUC
      Random Forest  0.706984   0.700595 0.671386  0.685680 0.763354
  Gradient Boosting  0.739468   0.757034 0.666667  0.708983 0.805611
Logistic Regression  0.734413   0.765345 0.637561  0.695634 0.798412
                SVM  0.736847   0.779774 0.623206  0.692753 0.787034
            XGBoost  0.741621   0.765836 0.658604  0.708184 0.804620

En Önemli 10 Feature - Outlier'lı Veriler (Random Forest):
  weight: 0.2308
  height: 0.2101
  ap_hi: 0.1931
  age_years: 0.1635
  ap_lo: 0.0912
  cholesterol: 0.0392
  gluc: 0.0195
  gender: 0.0185
  active: 0.0161
  smoke: 0.0097

En Önemli 10 Feature - Outlier'lar çıkarılmış (Random Forest):
  weight: 0.2491
  height: 0.2220
  ap_hi: 0.1939
  age_years: 0.1714
  ap_lo: 0.0776
  cholesterol: 0.0367
  gender: 0.0175
  active: 0.0148
  smoke: 0.0093
  alco: 0.0078

================================================================================
KARŞILAŞTIRMALI ANALİZ PIPELINE TAMAMLANDI!
================================================================================
Grafikler proje dizinine kaydedildi:
- confusion_matrices.png (her iki veri seti için)
- roc_curves.png (her iki veri seti için)
- metrics_comparison.png (her iki veri seti için)
- rf_feature_importance.png (her iki veri seti için)
- xgb_feature_importance.png (her iki veri seti için)
- comparative_analysis.png (karşılaştırmalı sonuçlar)
"""
Kalp Krizi Risk Tahmin Modeli - Optimizasyon Ana Çalıştırma Dosyası
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.advanced_feature_engineering import AdvancedFeatureEngineer
from analysis.hyperparameter_tuning import HyperparameterTuner
from analysis.ensemble_methods import EnsembleMethods
from utils.data_loader import DataLoader

def main():
    """Ana fonksiyon - Tüm optimizasyon pipeline'ını çalıştır."""
    print("="*80)
    print("KALP KRİZİ RİSK TAHMİN MODELİ - TAM OPTİMİZASYON PIPELINE")
    print("="*80)
    
    # 1. Veri dosyası yolu
    data_path = project_root / "data" / "cardiokaggle.csv"
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        return
    
    # 2. GELİŞMİŞ FEATURE ENGINEERING
    print("\n" + "="*50)
    print("1. GELİŞMİŞ FEATURE ENGINEERING")
    print("="*50)
    
    advanced_fe = AdvancedFeatureEngineer()
    advanced_data = advanced_fe.advanced_pipeline(str(data_path))
    
    if advanced_data is None:
        print("Gelişmiş feature engineering başarısız!")
        return
    
    print(f"Feature sayısı: {advanced_data['X_train'].shape[1]}")
    
    # 3. HYPERPARAMETER TUNING
    print("\n" + "="*50)
    print("2. HYPERPARAMETER TUNING")
    print("="*50)
    
    tuner = HyperparameterTuner()
    best_models = tuner.tune_all_models(
        advanced_data['X_train'],
        advanced_data['y_train'],
        cv=5,
        n_jobs=-1
    )
    
    # Tune edilmiş modelleri test setinde değerlendir
    tuning_results = tuner.evaluate_tuned_models(
        advanced_data['X_test'],
        advanced_data['y_test']
    )
    
    # 4. ENSEMBLE METHODS
    print("\n" + "="*50)
    print("3. ENSEMBLE METHODS")
    print("="*50)
    
    ensemble = EnsembleMethods()
    ensemble_results = ensemble.run_ensemble_analysis(
        advanced_data['X_train'],
        advanced_data['X_test'],
        advanced_data['y_train'],
        advanced_data['y_test']
    )
    
    # 5. FİNAL SONUÇLAR
    print("\n" + "="*80)
    print("FİNAL OPTİMİZASYON SONUÇLARI")
    print("="*80)
    
    best_tuning_model = max(tuning_results.keys(), key=lambda x: tuning_results[x]['accuracy'])
    best_tuning_accuracy = tuning_results[best_tuning_model]['accuracy']
    
    best_ensemble_model = ensemble_results['best_model_name']
    best_ensemble_accuracy = ensemble_results['best_accuracy']
    
    print(f"Hyperparameter Tuning (En iyi): {best_tuning_accuracy:.4f}")
    print(f"Ensemble Methods (En iyi): {best_ensemble_accuracy:.4f}")
    
    if best_ensemble_accuracy > best_tuning_accuracy:
        print(f"🏆 En iyi sonuç: {best_ensemble_model} ({best_ensemble_accuracy:.4f})")
    else:
        print(f"🏆 En iyi sonuç: {best_tuning_model} ({best_tuning_accuracy:.4f})")

if __name__ == "__main__":
    main()
"""
Kalp Krizi Risk Tahmin Modeli - Optimizasyon Ana Çalıştırma Dosyası
"""

import sys
from pathlib import Path

# Proje kök dizinini Python path'ine ekle
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.advanced_feature_engineering import AdvancedFeatureEngineer
from analysis.hyperparameter_tuning import HyperparameterTuner
from analysis.ensemble_methods import EnsembleMethods
from utils.data_loader import DataLoader

def main():
    """Ana fonksiyon - Tüm optimizasyon pipeline'ını çalıştır."""
    print("="*80)
    print("KALP KRİZİ RİSK TAHMİN MODELİ - TAM OPTİMİZASYON PIPELINE")
    print("="*80)
    
    # 1. Veri dosyası yolu
    data_path = project_root / "data" / "cardiokaggle.csv"
    
    if not data_path.exists():
        print(f"HATA: Veri dosyası bulunamadı: {data_path}")
        return
    
    # 2. GELİŞMİŞ FEATURE ENGINEERING
    print("\n" + "="*50)
    print("1. GELİŞMİŞ FEATURE ENGINEERING")
    print("="*50)
    
    advanced_fe = AdvancedFeatureEngineer()
    advanced_data = advanced_fe.advanced_pipeline(str(data_path))
    
    if advanced_data is None:
        print("Gelişmiş feature engineering başarısız!")
        return
    
    print(f"Feature sayısı: {advanced_data['X_train'].shape[1]}")
    
    # 3. HYPERPARAMETER TUNING
    print("\n" + "="*50)
    print("2. HYPERPARAMETER TUNING")
    print("="*50)
    
    tuner = HyperparameterTuner()
    best_models = tuner.tune_all_models(
        advanced_data['X_train'],
        advanced_data['y_train'],
        cv=5,
        n_jobs=-1
    )
    
    # Tune edilmiş modelleri test setinde değerlendir
    tuning_results = tuner.evaluate_tuned_models(
        advanced_data['X_test'],
        advanced_data['y_test']
    )
    
    # 4. ENSEMBLE METHODS
    print("\n" + "="*50)
    print("3. ENSEMBLE METHODS")
    print("="*50)
    
    ensemble = EnsembleMethods()
    ensemble_results = ensemble.run_ensemble_analysis(
        advanced_data['X_train'],
        advanced_data['X_test'],
        advanced_data['y_train'],
        advanced_data['y_test']
    )
    
    # 5. FİNAL SONUÇLAR
    print("\n" + "="*80)
    print("FİNAL OPTİMİZASYON SONUÇLARI")
    print("="*80)
    
    best_tuning_model = max(tuning_results.keys(), key=lambda x: tuning_results[x]['accuracy'])
    best_tuning_accuracy = tuning_results[best_tuning_model]['accuracy']
    
    best_ensemble_model = ensemble_results['best_model_name']
    best_ensemble_accuracy = ensemble_results['best_accuracy']
    
    print(f"Hyperparameter Tuning (En iyi): {best_tuning_accuracy:.4f}")
    print(f"Ensemble Methods (En iyi): {best_ensemble_accuracy:.4f}")
    
    if best_ensemble_accuracy > best_tuning_accuracy:
        print(f"🏆 En iyi sonuç: {best_ensemble_model} ({best_ensemble_accuracy:.4f})")
    else:
        print(f"🏆 En iyi sonuç: {best_tuning_model} ({best_tuning_accuracy:.4f})")

if __name__ == "__main__":
    main()