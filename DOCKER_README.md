# Cardiyovask Docker Kurulumu

Bu proje Docker kullanarak containerize edilmiştir. Aşağıdaki adımları takip ederek projeyi çalıştırabilirsiniz.

## Gereksinimler

- Docker
- Docker Compose

## Hızlı Başlangıç

1. **Environment dosyasını oluşturun:**
   ```bash
   cp env.example .env
   ```

2. **Environment değişkenlerini düzenleyin:**
   `.env` dosyasını açın ve gerekli değerleri değiştirin.

3. **Projeyi başlatın:**
   ```bash
   docker-compose up -d
   ```

4. **Uygulamaya erişin:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Dokümantasyonu: http://localhost:8000/docs

## Servisler

### Backend API (Port: 8000)
- FastAPI tabanlı REST API
- PostgreSQL veritabanı bağlantısı
- JWT authentication
- Machine Learning modelleri

### Frontend (Port: 3000)
- React tabanlı web arayüzü
- Nginx ile serve edilir
- Backend API'ye proxy yapılır

### PostgreSQL (Port: 5432)
- Veritabanı servisi
- Otomatik tablo oluşturma

## Environment Değişkenleri

### Veritabanı
- `DB_HOST`: Veritabanı host adresi
- `DB_PORT`: Veritabanı portu
- `DB_NAME`: Veritabanı adı
- `DB_USER`: Veritabanı kullanıcı adı
- `DB_PASSWORD`: Veritabanı şifresi

### API
- `API_HOST`: API host adresi
- `API_PORT`: API portu
- `DEBUG`: Debug modu (True/False)

### JWT
- `SECRET_KEY`: JWT secret key
- `ALGORITHM`: JWT algoritması
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Token süresi

### CORS
- `CORS_ORIGINS`: İzin verilen origin'ler
- `CORS_ALLOW_CREDENTIALS`: Credential desteği
- `CORS_ALLOW_METHODS`: İzin verilen HTTP metodları
- `CORS_ALLOW_HEADERS`: İzin verilen header'lar

### Frontend
- `FRONTEND_PORT`: Frontend portu
- `REACT_APP_API_BASE_URL`: Backend API URL'i

### Model
- `MODELS_DIR`: Model dosyaları dizini
- `DATA_FILE`: Veri dosyası yolu
- `USE_GPU`: GPU kullanımı
- `GPU_DEVICE_ID`: GPU cihaz ID'si

## Docker Komutları

### Tüm servisleri başlat
```bash
docker-compose up -d
```

### Servisleri durdur
```bash
docker-compose down
```

### Logları görüntüle
```bash
docker-compose logs -f
```

### Belirli bir servisin loglarını görüntüle
```bash
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f db
```

### Servisleri yeniden build et
```bash
docker-compose up --build -d
```

### Veritabanını sıfırla
```bash
docker-compose down -v
docker-compose up -d
```

## Geliştirme

### Backend geliştirme
```bash
# Backend container'ına bağlan
docker-compose exec backend bash

# Python bağımlılıklarını yükle
pip install -r requirements.txt

# API'yi çalıştır
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend geliştirme
```bash
# Frontend container'ına bağlan
docker-compose exec frontend sh

# Bağımlılıkları yükle
npm install

# Development server'ı başlat
npm start
```

## Sorun Giderme

### Port çakışması
Eğer portlar kullanımda ise, `.env` dosyasında farklı portlar belirleyin:
```
API_PORT=8001
FRONTEND_PORT=3001
DB_PORT=5433
```

### Veritabanı bağlantı hatası
```bash
# Veritabanı container'ının çalıştığını kontrol et
docker-compose ps

# Veritabanı loglarını kontrol et
docker-compose logs db
```

### Model dosyaları bulunamıyor
Model dosyalarının `models/` dizininde olduğundan emin olun.

## Production Deployment

Production ortamında:
1. `.env` dosyasında güvenli değerler kullanın
2. `SECRET_KEY`'i güçlü bir değerle değiştirin
3. `CORS_ORIGINS`'i spesifik domainlerle sınırlayın
4. `DEBUG=False` yapın
5. SSL sertifikası ekleyin

## Backup ve Restore

### Veritabanı backup
```bash
docker-compose exec db pg_dump -U postgres cardiyovask_db > backup.sql
```

### Veritabanı restore
```bash
docker-compose exec -T db psql -U postgres cardiyovask_db < backup.sql
```
