#!/usr/bin/env python3
"""
Randevu atama işlemini çalıştır
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from postgres_appointment_system import manual_assign_appointments

if __name__ == "__main__":
    print("Randevu atama islemi baslatiliyor...")
    manual_assign_appointments()
    print("Islem tamamlandi!")