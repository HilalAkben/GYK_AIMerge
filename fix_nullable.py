#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.core.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # patient_id ve doctor_id kolonlarını nullable yap
    db.execute(text('ALTER TABLE appointments ALTER COLUMN patient_id DROP NOT NULL'))
    db.execute(text('ALTER TABLE appointments ALTER COLUMN doctor_id DROP NOT NULL'))
    db.commit()
    print('✅ patient_id ve doctor_id kolonları nullable yapıldı')
except Exception as e:
    print(f'❌ Hata: {e}')
finally:
    db.close()
