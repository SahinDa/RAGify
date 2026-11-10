from typing import Optional, Union
from pydantic import BaseModel, Field
from fastapi import Depends, Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User, UserRole
from app.auth.security import decode_access_token

security = HTTPBearer(auto_error=False)


class DemoUser(BaseModel):
    """Virtual user object representing an unauthenticated guest."""
    id: str = "demo-guest"
    email: str = "guest@demo.local"
    first_name: str = "Guest"
    last_name: Optional[str] = None
    role: int = Field(default_factory=lambda: int(UserRole.USER))
    is_demo: bool = True


def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Union[User, DemoUser]:
    """Resolves authenticated User from cookie/bearer header, or defaults to DemoUser."""
    token = request.cookies.get("ragify_token")
    if not token and credentials:
        token = credentials.credentials

    if not token:
        return DemoUser()

    payload = decode_access_token(token)
    if not payload:
        return DemoUser()

    user_id = payload.get("sub")
    if not user_id:
        return DemoUser()

    user = db.query(User).filter(User.id == user_id).first()
    return user if user else DemoUser()


get_current_user_or_guest = get_optional_user


def get_current_user(
    user: Union[User, DemoUser] = Depends(get_optional_user),
) -> User:
    """Strict check: raises 401 if the user is a Demo / Guest."""
    if isinstance(user, DemoUser) or getattr(user, "is_demo", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in or create an account.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Enforces that the authenticated user possesses admin privileges."""
    if int(current_user.role) != int(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operation requires administrative privileges",
        )
    return current_user