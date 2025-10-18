from api.core.database import SessionLocal, ModelRecord, PredictionRecord

db = SessionLocal()

print('=== ModelRecord tablosundaki modeller ===')
models = db.query(ModelRecord).all()
for model in models:
    print(f'ID: {model.id}, Name: {model.model_name}, Active: {model.is_active}')

print('\n=== PredictionRecord tablosundaki model_id\'ler ===')
predictions = db.query(PredictionRecord).distinct(PredictionRecord.model_id).all()
for pred in predictions:
    print(f'Model ID: {pred.model_id}')

db.close()
