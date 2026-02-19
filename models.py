from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from database import Base

class TransactionLog(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    result = Column(String)
    risk_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
