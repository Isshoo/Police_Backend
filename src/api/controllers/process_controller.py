from flask import request, jsonify
from src.api.services.process_service import ProcessService
from src.api.services.preprocess_service import PreprocessService


class ProcessController:
    process_service = ProcessService()
    preprocess_service = PreprocessService()

    def __init__(self):
        pass

    def split_dataset(self, preprocessed_dataset_id):
        try:
            data = request.json
            if "test_size" not in data or "raw_dataset_id" not in data:
                return jsonify({"error": "Invalid request"}), 400

            test_size = data["test_size"]
            raw_dataset_id = data["raw_dataset_id"]

            if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
                return jsonify({"error": "Test size must be a positive float between 0 and 1"}), 400

            preprocessed_datasets = self.preprocess_service.fetch_preprocessed_datasets(
                raw_dataset_id)
            if not preprocessed_datasets:
                return jsonify({"error": "No preprocessed datasets found"}), 404

            preprocessed_dataset_path = next(
                (d["path"] for d in preprocessed_datasets if d["id"] == preprocessed_dataset_id), None)
            if not preprocessed_dataset_path:
                return jsonify({"error": "Preprocessed dataset not found"}), 404

            result = self.process_service.split_dataset(
                preprocessed_dataset_path, test_size)
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def train_model(self, preprocessed_dataset_id):
        try:
            data = request.json
            required_keys = ["raw_dataset_id",
                             "name", "n_neighbors", "test_size"]
            if not all(key in data for key in required_keys):
                return jsonify({"error": "Invalid request"}), 400

            raw_dataset_id = data["raw_dataset_id"]
            name = data["name"]
            n_neighbors = data["n_neighbors"]
            test_size = data["test_size"]

            if not isinstance(n_neighbors, int) or n_neighbors <= 0:
                return jsonify({"error": "Number of neighbors must be a positive integer"}), 400
            if not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1:
                return jsonify({"error": "Test size must be a positive float between 0 and 1"}), 400

            preprocessed_datasets = self.preprocess_service.fetch_preprocessed_datasets(
                raw_dataset_id)
            if not preprocessed_datasets:
                return jsonify({"error": "No preprocessed datasets found"}), 404
            preprocessed_dataset_path = next(
                (d["path"] for d in preprocessed_datasets if d["id"] == preprocessed_dataset_id), None)
            if not preprocessed_dataset_path:
                return jsonify({"error": "Preprocessed dataset not found"}), 404

            # cek nama
            if any(d["name"] == name for d in self.process_service.get_models()):
                return jsonify({"error": "Model name already exists"}), 400

            model_metadata = self.process_service.train_model(
                preprocessed_dataset_id, preprocessed_dataset_path, raw_dataset_id, name, n_neighbors, test_size
            )
            return jsonify(model_metadata)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def get_models(self):
        try:
            result = self.process_service.get_models()
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def get_model(self, model_id):
        try:
            result = self.process_service.get_model(model_id)
            if not result:
                return jsonify({"error": "Model not found"}), 404
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def edit_model_name(self, model_id):
        try:
            if model_id == 'default-stemmed':
                return jsonify({"error": "Default model cannot be edited"}), 400

            data = request.json
            if "new_name" not in data:
                return jsonify({"error": "Invalid request"}), 400

            new_name = data["new_name"]
            if not isinstance(new_name, str) or len(new_name) < 3:
                return jsonify({"error": "New name must be a string with at least 3 characters"}), 400
            # cek nama
            if any(d["name"] == new_name for d in self.process_service.get_models() if d["id"] != model_id):
                return jsonify({"error": "Model name already exists"}), 400

            success = self.process_service.edit_model_name(
                model_id, new_name)
            if success:
                return jsonify({"message": "Model name updated successfully"})
            return jsonify({"error": "Model not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def delete_model(self, model_id):
        try:
            if model_id == 'default-stemmed':
                return jsonify({"error": "Default model cannot be deleted"}), 400

            success = self.process_service.delete_model(model_id)
            if success:
                return jsonify({"message": "Model deleted successfully"})
            return jsonify({"error": "Model not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def get_model_evaluation(self, model_id):
        try:
            result = self.process_service.get_evaluation(model_id)
            if not result:
                return jsonify({"error": "Evaluation not found"}), 404
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def get_model_parameters(self, model_id):
        try:
            result = self.process_service.get_parameters(model_id)
            if not result:
                return jsonify({"error": "Parameters not found"}), 404
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def fetch_tfidf_stats(self, model_id):
        try:
            page = int(request.args.get("page", 1))
            limit = int(request.args.get("limit", 10))
            result = self.process_service.tfidf_stats(
                model_id, page, limit)
            if not result:
                return jsonify({"error": "TFIDF stats not found"}), 404
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def fetch_neighbors(self, model_id):
        try:
            page = int(request.args.get("page", 1))
            limit = int(request.args.get("limit", 10))
            result = self.process_service.neighbors(
                model_id, page, limit)
            if not result:
                return jsonify({"error": "Neighbors not found"}), 404
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
