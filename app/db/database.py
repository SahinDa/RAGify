import os
from dotenv import load_dotenv
load_dotenv()
import asyncpg
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Read directly from environment variable
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Provides a transactional database session per request and closes it after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_db_pool() -> asyncpg.Pool:
    """Returns a shared asyncpg connection pool, initializing it lazily."""
    global _async_pool
    if _async_pool is None:
        _async_pool = await asyncpg.create_pool(
            dsn=DATABASE_URL,
            min_size=1,
            max_size=5,
            command_timeout=30,
        )
    return _async_pool


async def close_db_pool() -> None:
    """Closes the shared asyncpg pool cleanly on application shutdown."""
    global _async_pool
    if _async_pool is not None:
        await _async_pool.close()
        _async_pool = None        