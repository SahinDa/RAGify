import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, UserRole
from app.auth.schemas import UserSignUp, UserLogin, UserResponse, AuthResponse
from app.auth.security import hash_password, verify_password, create_access_token
from app.auth.deps import get_current_user_or_guest, DemoUser

router = APIRouter(prefix="/auth", tags=["Authentication"])

COOKIE_MAX_AGE = 24 * 60 * 60  # 24 Hours
IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key="ragify_token",
        value=token,
        max_age=COOKIE_MAX_AGE,
        expires=COOKIE_MAX_AGE,
        httponly=False,  # Set to False to permit frontend fallback inspection
        samesite="lax",
        secure=IS_PRODUCTION,
        path="/",
    )


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(data: UserSignUp, response: Response, db: Session = Depends(get_db)):
    """Register a new user account."""
    normalized_email = data.email.lower().strip()
    existing_user = db.query(User).filter(User.email == normalized_email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists.",
        )

    first_name = data.first_name.strip()
    if not first_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="First name cannot be empty.",
        )

    last_name = data.last_name.strip() if data.last_name and data.last_name.strip() else None

    new_user = User(
        email=normalized_email,
        hashed_password=hash_password(data.password),
        first_name=first_name,
        last_name=last_name,
        role=UserRole.USER,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(
        data={"sub": str(new_user.id), "role": int(new_user.role), "email": new_user.email}
    )
    set_auth_cookie(response, token)

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        expires_in=COOKIE_MAX_AGE,
        user=UserResponse(
            id=str(new_user.id),
            email=new_user.email,
            first_name=new_user.first_name,
            last_name=new_user.last_name,
            role=int(new_user.role),
            created_at=new_user.created_at,
        ),
    )


@router.post("/login", response_model=AuthResponse)
def login(credentials: UserLogin, response: Response, db: Session = Depends(get_db)):
    """Authenticate email & password and issue JWT session."""
    normalized_email = credentials.email.lower().strip()
    user = db.query(User).filter(User.email == normalized_email).first()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={"sub": str(user.id), "role": int(user.role), "email": user.email}
    )
    set_auth_cookie(response, token)

    return AuthResponse(
        access_token=token,
        token_type="bearer",
        expires_in=COOKIE_MAX_AGE,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            role=int(user.role),
            created_at=user.created_at,
        ),
    )


@router.post("/logout", status_code=status.HTTP_200_OK)
def logout(response: Response):
    """Terminate user session and clear cookie."""
    response.delete_cookie(key="ragify_token", path="/")
    return {"message": "Successfully logged out", "status": "success"}


@router.get("/me")
def get_me(user=Depends(get_current_user_or_guest)):
    """Returns active user session; returns is_demo: True for guests without 401 errors."""
    if isinstance(user, DemoUser) or getattr(user, "is_demo", False):
        return {
            "authenticated": False,
            "is_demo": True,
            "user": None
        }

    return {
        "authenticated": True,
        "is_demo": False,
        "user": UserResponse(
            id=str(user.id),
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            role=int(user.role),
            created_at=user.created_at,
        )
    }