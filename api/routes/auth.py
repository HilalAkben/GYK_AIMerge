"""
Kimlik Doğrulama API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from api.core.database import get_db, User
from api.core.config import settings
from api.services.auth_service import (
    authenticate_user, 
    create_access_token, 
    get_password_hash,
    get_current_active_user,
    require_admin
)
from pydantic import BaseModel

router = APIRouter()

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: str
    role: str = "user"

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
    role: str
    is_active: bool

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

@router.post("/register", response_model=dict)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Kullanıcı kayıt."""
    # Kullanıcı adı kontrolü
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=400,
            detail="Kullanıcı adı zaten kullanılıyor"
        )
    
    # Email kontrolü
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=400,
            detail="Email adresi zaten kullanılıyor"
        )
    
    # Yeni kullanıcı oluştur
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {
        "message": "Kullanıcı başarıyla oluşturuldu",
        "user_id": db_user.id,
        "username": db_user.username
    }

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Kullanıcı giriş."""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yanlış kullanıcı adı veya şifre",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, 
        expires_delta=access_token_expires
    )
    
    user_response = UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active
    )
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": user_response
    }

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Mevcut kullanıcı bilgilerini al."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active
    )

@router.post("/create-admin", response_model=dict)
async def create_admin_user(user_data: UserCreate, db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    """Admin kullanıcısı oluştur (sadece admin yetkisi ile)."""
    # Kullanıcı adı kontrolü
    if db.query(User).filter(User.username == user_data.username).first():
        raise HTTPException(
            status_code=400,
            detail="Kullanıcı adı zaten kullanılıyor"
        )
    
    # Email kontrolü
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(
            status_code=400,
            detail="Email adresi zaten kullanılıyor"
        )
    
    # Admin kullanıcısı oluştur
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
        role="admin"  # Admin rolü zorla
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return {
        "message": "Admin kullanıcısı başarıyla oluşturuldu",
        "user_id": db_user.id,
        "username": db_user.username,
        "role": db_user.role
    }

@router.get("/users", response_model=list[UserResponse])
async def list_users(db: Session = Depends(get_db), current_user: User = Depends(require_admin)):
    """Tüm kullanıcıları listele (sadece admin)."""
    users = db.query(User).all()
    return [
        UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active
        ) for user in users
    ]

@router.post("/logout")
async def logout():
    """Çıkış yapma endpoint'i."""
    return {
        "success": True,
        "message": "Başarıyla çıkış yapıldı"
    }

