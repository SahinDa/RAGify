# app/db/models.py
import uuid
from datetime import datetime
from enum import IntEnum
from sqlalchemy import Column, String,Integer, DateTime, Enum as SQLEnum
from app.db.database import Base  # or declarative_base()


class UserRole(IntEnum):
    USER = 1
    ADMIN = 2


class User(Base):
    __tablename__ = "users"

    # Maps directly to your Neon table columns
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Integer, default=UserRole.USER.value, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

from app.db.models import UserRole    