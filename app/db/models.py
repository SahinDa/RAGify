# app/db/models.py
import uuid
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, String, DateTime, Enum as SQLEnum
from app.database import Base  # or declarative_base()


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"


class User(Base):
    __tablename__ = "users"

    # Maps directly to your Neon table columns
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(
        SQLEnum(UserRole, name="user_roles", create_type=False),
        default=UserRole.USER,
        nullable=False,
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)