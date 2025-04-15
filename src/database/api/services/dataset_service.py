import uuid
from datetime import datetime
from src.database.config import SessionLocal
from src.database.models.dataset import Dataset
import pandas as pd


class DatasetService:
    def fetch_all(self):
        session = SessionLocal()
        try:
            datasets = session.query(Dataset).all()
            return [d.to_dict() for d in datasets]
        finally:
            session.close()

    def fetch_by_id(self, dataset_id):
        session = SessionLocal()
        try:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            return dataset.to_dict() if dataset else None
        finally:
            session.close()

    def create(self, data):
        session = SessionLocal()
        try:
            df = pd.read_csv(data["path"])

            total_data = len(df)
            label_counts = df['label'].value_counts(
            ).to_dict() if 'label' in df.columns else {}

            new_dataset = Dataset(
                id=str(uuid.uuid4()),
                name=data.get("name"),
                path=data.get("path"),
                total_data=total_data,
                label_counts=label_counts,
                upload_at=datetime.utcnow()
            )
            session.add(new_dataset)
            session.commit()
            return new_dataset.to_dict()
        finally:
            session.close()

    def delete(self, dataset_id):
        session = SessionLocal()
        try:
            dataset = session.query(Dataset).filter_by(id=dataset_id).first()
            if not dataset:
                return False
            session.delete(dataset)
            session.commit()
            return True
        finally:
            session.close()
