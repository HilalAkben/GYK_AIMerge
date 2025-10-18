"""
Kimlik Doğrulama Servisi
"""

from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
import hashlib
import secrets
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from api.core.config import settings
from api.core.database import get_db, User

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Şifreyi doğrula."""
    # Hash'den salt'ı çıkar
    if ':' not in hashed_password:
        return False
    salt, hash_part = hashed_password.split(':', 1)
    # Şifreyi hashle ve karşılaştır
    password_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
    return password_hash == hash_part

def get_password_hash(password: str) -> str:
    """Şifreyi hashle."""
    # Salt oluştur
    salt = secrets.token_hex(16)
    # Şifre + salt'ı hashle
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    # Salt:hash formatında döndür
    return f"{salt}:{password_hash}"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """JWT token oluştur."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def authenticate_user(db: Session, username: str, password: str):
    """Kullanıcıyı doğrula."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Mevcut kullanıcıyı al."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token doğrulanamadı",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Aktif kullanıcıyı al."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Pasif kullanıcı")
    return current_user

def require_admin(current_user: User = Depends(get_current_active_user)):
    """Admin yetkisi gerektirir."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin yetkisi gerekli"
        )
    return current_user

def require_doctor(current_user: User = Depends(get_current_active_user)):
    """Doctor yetkisi gerektirir."""
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor yetkisi gerekli"
        )
    return current_user

def require_doctor_or_admin(current_user: User = Depends(get_current_active_user)):
    """Doctor veya Admin yetkisi gerektirir."""
    if current_user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor veya Admin yetkisi gerekli"
        )
    return current_user

def require_patient(current_user: User = Depends(get_current_active_user)):
    """Patient (user) yetkisi gerektirir."""
    if current_user.role != "user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hasta yetkisi gerekli"
        )
    return current_user

def require_any_role(current_user: User = Depends(get_current_active_user)):
    """Herhangi bir geçerli role yetkisi gerektirir."""
    if current_user.role not in ["user", "doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Geçerli kullanıcı yetkisi gerekli"
        )
    return current_user
