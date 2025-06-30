import base64
import phpserialize
from sqlalchemy.orm import Session
from models.session import UserSession

def get_user_payload(session_id: str, db: Session) -> dict:
    session = db.query(UserSession).filter(UserSession.id == session_id).first()
    if not session:
        raise ValueError("Session not found in database.")

    try:
        decoded_payload = base64.b64decode(session.payload)
        deserialized = phpserialize.loads(decoded_payload, decode_strings=True)
    except Exception as e:
        raise ValueError(f"Failed to decode Laravel payload: {e}")

    return deserialized