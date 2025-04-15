from flask import jsonify, request
import uuid
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from src.database.api.services.dataset_service import DatasetService


class DatasetController:
    def __init__(self):
        self.service = DatasetService()
        self.UPLOAD_FOLDER = "src/storage/datasets/uploads/db"

    def get_datasets(self):
        datasets = self.service.fetch_all()
        return jsonify(datasets)

    def get_dataset_by_id(self, dataset_id):
        dataset = self.service.fetch_by_id(dataset_id)
        if dataset:
            return jsonify(dataset)
        return jsonify({"message": "Dataset not found"}), 404

    def create_dataset(self):
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(self.UPLOAD_FOLDER, filename)
        file.save(file_path)

        # kirim data ke service
        data = {
            "name": filename,
            "path": file_path
        }
        result = self.service.create(data)
        return jsonify(result), 201

    def delete_dataset(self, dataset_id):
        success = self.service.delete(dataset_id)
        if success:
            return jsonify({"message": "Deleted successfully"})
        return jsonify({"message": "Dataset not found"}), 404
