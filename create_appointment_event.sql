-- MySQL EVENT ile otomatik randevu atama sistemi
-- Her 10 dakikada bir çalışacak

-- Önce mevcut event'i temizle
DROP EVENT IF EXISTS give_appointments;

-- Event oluştur
DELIMITER $$

CREATE EVENT IF NOT EXISTS give_appointments
ON SCHEDULE EVERY 10 MINUTE
DO
BEGIN
  -- Mevcut randevuları temizle (her seferinde yeniden oluştur)
  DELETE FROM appointments;
  
  -- prediction_records tablosundaki risk_score'a göre en yüksek 20 hastayı seç ve appointments tablosuna ekle
  INSERT INTO appointments (
    prediction_record_id,
    age,
    gender,
    height,
    weight,
    ap_hi,
    ap_lo,
    cholesterol,
    gluc,
    smoke,
    alco,
    active,
    risk_score,
    priority_score,
    prediction_label,
    prediction_probability,
    model_name,
    status,
    priority,
    queue_position,
    appointment_date,
    assigned_at,
    created_at,
    updated_at
  )
  SELECT
    pr.id AS prediction_record_id,
    pr.age,
    pr.gender,
    pr.height,
    pr.weight,
    pr.ap_hi,
    pr.ap_lo,
    pr.cholesterol,
    pr.gluc,
    pr.smoke,
    pr.alco,
    pr.active,
    pr.risk_score,
    pr.risk_score AS priority_score, -- risk_score'u priority_score olarak kullan
    pr.prediction AS prediction_label,
    pr.probability AS prediction_probability,
    CONCAT('Model_', pr.model_id) AS model_name, -- Model adı oluştur
    'pending' AS status,
    CASE
      WHEN pr.risk_score >= 0.8 THEN 'urgent'
      WHEN pr.risk_score >= 0.6 THEN 'high'
      WHEN pr.risk_score >= 0.4 THEN 'normal'
      ELSE 'low'
    END AS priority,
    ROW_NUMBER() OVER (ORDER BY pr.risk_score DESC) AS queue_position,
    DATE_ADD(NOW(), INTERVAL 1 DAY) AS appointment_date, -- randevu tarihi = yarın
    NOW() AS assigned_at,
    NOW() AS created_at,
    NOW() AS updated_at
  FROM prediction_records pr
  ORDER BY pr.risk_score DESC
  LIMIT 20;

  -- Log için kayıt sayısını yazdır
  SELECT CONCAT('✅ ', COUNT(*), ' hasta randevu tablosuna atandı') AS result
  FROM appointments;
  
END$$

DELIMITER ;

-- Event scheduler'ı aktif et
SET GLOBAL event_scheduler = ON;

-- Event'leri göster
SHOW EVENTS;

-- Test için manuel çalıştır
-- CALL give_appointments();
