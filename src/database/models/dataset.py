from sqlalchemy import Column, String, Integer, DateTime, JSON
from src.database.config import Base
from datetime import datetime


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    path = Column(String, nullable=False)
    total_data = Column(Integer, default=0)
    label_counts = Column(JSON, default={})
    upload_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "total_data": self.total_data,
            "label_counts": self.label_counts,
            "upload_at": self.upload_at,
        }
