#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # Appointment sayısı
    result = db.execute(text('SELECT COUNT(*) FROM appointments'))
    app_count = result.scalar()
    print(f'Appointments: {app_count}')
    
    # PredictionRecord sayısı  
    result = db.execute(text('SELECT COUNT(*) FROM prediction_records'))
    pred_count = result.scalar()
    print(f'PredictionRecords: {pred_count}')
    
    if app_count > 0:
        result = db.execute(text('SELECT id, age, risk_score, priority FROM appointments LIMIT 3'))
        for row in result:
            print(f'  ID:{row[0]} Yaş:{row[1]} Risk:{row[2]:.3f} Öncelik:{row[3]}')
finally:
    db.close()
