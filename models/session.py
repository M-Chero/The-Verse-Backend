from sqlalchemy import Column, String, Text, Integer
from database import Base

class UserSession(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=True)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    payload = Column(Text)
    last_activity = Column(Integer, index=True)