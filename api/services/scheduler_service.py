"""
Scheduled Jobs Service
Günlük otomatik sıralama ve diğer zamanlanmış görevler için.
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session
from datetime import datetime
import logging
from typing import Dict, Any

from api.core.database import SessionLocal, PredictionRecord
from api.services.prediction_service import PredictionService

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchedulerService:
    """Zamanlanmış görevler için servis."""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.prediction_service = PredictionService()
        
    def start(self):
        """Scheduler'ı başlat."""
        try:
            # Her gün 23:59'da otomatik sıralama yap
            self.scheduler.add_job(
                func=self.daily_auto_sorting,
                trigger=CronTrigger(hour=23, minute=59),
                id='daily_auto_sorting',
                name='Günlük Otomatik Sıralama',
                replace_existing=True
            )
            
            # Scheduler'ı başlat
            self.scheduler.start()
            logger.info("Scheduler başlatıldı - Günlük otomatik sıralama 23:59'da çalışacak")
            
        except Exception as e:
            logger.error(f"Scheduler başlatma hatası: {str(e)}")
    
    def stop(self):
        """Scheduler'ı durdur."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler durduruldu")
    
    async def daily_auto_sorting(self):
        """Her gün 23:59'da çalışacak otomatik sıralama."""
        try:
            logger.info("Günlük otomatik sıralama başlatıldı...")
            
            db = SessionLocal()
            try:
                # Tüm prediction kayıtlarını getir
                records = db.query(PredictionRecord).all()
                
                if not records:
                    logger.info("Sıralanacak kayıt bulunamadı")
                    return
                
                # Risk skorlarına göre sırala (yüksek risk önce)
                # Aynı risk skorunda olanlar için tarihe göre sırala (yeni olan önce)
                sorted_records = sorted(records, key=lambda x: (
                    -x.risk_score,  # Risk skoru yüksekten düşüğe
                    -x.created_at.timestamp() if x.created_at else 0  # Tarih yeniye göre
                ))
                
                # Sıralama sonuçlarını logla
                logger.info(f"Toplam {len(sorted_records)} kayıt sıralandı:")
                
                for i, record in enumerate(sorted_records[:10]):  # İlk 10 kaydı logla
                    logger.info(f"  #{i+1}: ID={record.id}, Risk={record.risk_score:.3f}, Tarih={record.created_at}")
                
                # Veritabanında güncelleme yapmak isterseniz burada yapabilirsiniz
                # Örneğin priority_order alanı ekleyip güncelleyebiliriz
                
                logger.info("Günlük otomatik sıralama tamamlandı")
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Günlük otomatik sıralama hatası: {str(e)}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Scheduler durumunu döndür."""
        jobs = []
        if self.scheduler.running:
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
        
        return {
            "running": self.scheduler.running,
            "jobs": jobs
        }
    
    async def manual_sorting(self) -> Dict[str, Any]:
        """Manuel sıralama - test için."""
        try:
            logger.info("Manuel sıralama başlatıldı...")
            await self.daily_auto_sorting()
            return {
                "success": True,
                "message": "Manuel sıralama başarıyla tamamlandı",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Manuel sıralama hatası: {str(e)}")
            return {
                "success": False,
                "message": f"Manuel sıralama hatası: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

# Global scheduler instance
scheduler_service = SchedulerService()
