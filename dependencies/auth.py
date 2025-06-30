import hashlib
import hmac
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dependencies.db import get_db
from sqlalchemy import text

security = HTTPBearer()

def verify_sanctum_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db=Depends(get_db),
) -> int:
    raw = credentials.credentials
    try:
        token_id_str, plain = raw.split("|", 1)
        token_id = int(token_id_str)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Malformed auth token")

    row = db.execute(
        text("""
            SELECT token, tokenable_id
            FROM personal_access_tokens
            WHERE id = :id
        """), {"id": token_id}
    ).first()

    if not row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token not found")

    hashed, user_id = row
    digest = hashlib.sha256(plain.encode()).hexdigest()
    if not hmac.compare_digest(digest, hashed):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return user_id