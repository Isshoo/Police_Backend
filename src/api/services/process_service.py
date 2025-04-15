import os
import json
import uuid
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from src.processing.trainer import KNNModelTrainer


class ProcessService:
    STORAGE_PATH = "src/storage/models/trained/"
    METADATA_PATH = "src/storage/metadatas/models.json"

    def __init__(self):
        os.makedirs(self.STORAGE_PATH, exist_ok=True)
        if not os.path.exists(self.METADATA_PATH):
            with open(self.METADATA_PATH, "w") as f:
                json.dump([], f)

    def load_metadata(self):
        with open(self.METADATA_PATH, "r") as f:
            return json.load(f)

    def save_metadata(self, metadata):
        with open(self.METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)

    def split_dataset(self, preprocessed_dataset_path, test_size):
        if not os.path.exists(preprocessed_dataset_path):
            return {}

        df = pd.read_csv(preprocessed_dataset_path, sep=",")
        if df.empty:
            return {}

        X = df["komentar"]
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        return {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_per_label": y_train.value_counts().to_dict(),
            "test_per_label": y_test.value_counts().to_dict()
        }

    def train_model(self, preprocessed_dataset_id, preprocessed_dataset_path, raw_dataset_id, name, n_neighbors, split_size):
        if not os.path.exists(preprocessed_dataset_path):
            return {}

        df = pd.read_csv(preprocessed_dataset_path, sep=",")
        if df.empty:
            return {}

        trainer = KNNModelTrainer(preprocessed_dataset_path)
        knn_model, evaluation_results, tfidf_stats, df_neighbors = trainer.train(
            n_neighbors, split_size)
        split_results = self.split_dataset(
            preprocessed_dataset_path, split_size)

        model_id = str(uuid.uuid4())
        model_path = os.path.join(self.STORAGE_PATH, f"{model_id}.joblib")
        joblib.dump(knn_model, model_path)  # Simpan model

        # Simpan metadata umum ke models.json
        metadata = self.load_metadata()
        model_metadata = {
            "id": model_id,
            "name": name,
            "model_path": model_path,
            "preprocessed_dataset_id": preprocessed_dataset_id,
            "raw_dataset_id": raw_dataset_id,
            "total_data": len(df),
            "n_neighbors": n_neighbors,
            "split_size": split_size,
            "accuracy": evaluation_results["accuracy"],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        metadata.append(model_metadata)
        self.save_metadata(metadata)

        # Simpan metadata tambahan per model
        model_meta_dir = os.path.join("src/storage/metadatas/models", model_id)
        os.makedirs(model_meta_dir, exist_ok=True)

        # Parameter
        parameters = {
            "n_neighbors": n_neighbors,
            "split_size": split_size,
            "train_size": split_results["train_size"],
            "test_size": split_results["test_size"],
            "train_per_label": split_results["train_per_label"],
            "test_per_label": split_results["test_per_label"]
        }
        with open(os.path.join(model_meta_dir, "parameters.json"), "w") as f:
            json.dump(parameters, f, indent=4)

        # Evaluation
        with open(os.path.join(model_meta_dir, "evaluation.json"), "w") as f:
            json.dump(evaluation_results, f, indent=4)

        # TF-IDF Stats
        tfidf_stats.to_csv(os.path.join(
            model_meta_dir, "tfidf_stats.csv"), index=False)

        # Neighbors
        df_neighbors.to_csv(os.path.join(
            model_meta_dir, "neighbors.csv"), index=False)

        return model_metadata

    def edit_model_name(self, model_id, new_name):
        metadata = self.load_metadata()
        model_metadata = next(
            (m for m in metadata if m["id"] == model_id), None)
        if not model_metadata:
            return False
        model_metadata["name"] = new_name
        model_metadata["updated_at"] = datetime.now().isoformat()
        self.save_metadata(metadata)
        return True

    def delete_model(self, model_id):
        metadata = self.load_metadata()
        model_metadata = next(
            (m for m in metadata if m["id"] == model_id), None)
        if not model_metadata:
            return False

        if model_id == 'default-stemmed':
            return False  # Default model cannot be deleted

        # Hapus file model jika ada
        model_path = model_metadata.get("model_path")
        if model_path and os.path.exists(model_path):
            os.remove(model_path)

        metadata = [m for m in metadata if m["id"] != model_id]
        self.save_metadata(metadata)
        return True

    def get_models(self):
        return self.load_metadata()

    def get_model(self, model_id):
        metadata = self.load_metadata()
        return next((m for m in metadata if m["id"] == model_id), {})

    def tfidf_stats(self, model_id, page=1, limit=10):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        file_path = os.path.join(model_dir, "tfidf_stats.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].to_dict(orient="records"),
            "total_data": len(df),
            "total_pages": (len(df) + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
        }

    def neighbors(self, model_id, page=1, limit=10):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        file_path = os.path.join(model_dir, "neighbors.csv")
        if not os.path.exists(file_path):
            return None

        df = pd.read_csv(file_path)
        start = (page - 1) * limit
        end = start + limit

        return {
            "data": df.iloc[start:end].to_dict(orient="records"),
            "total_data": len(df),
            "total_pages": (len(df) + limit - 1) // limit,
            "current_page": page,
            "limit": limit,
        }

    def get_parameters(self, model_id):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        param_path = os.path.join(model_dir, "parameters.json")
        if not os.path.exists(param_path):
            return None
        with open(param_path, "r") as f:
            return json.load(f)

    def get_evaluation(self, model_id):
        model_dir = os.path.join("src/storage/metadatas/models", model_id)
        eval_path = os.path.join(model_dir, "evaluation.json")
        if not os.path.exists(eval_path):
            return None
        with open(eval_path, "r") as f:
            return json.load(f)
