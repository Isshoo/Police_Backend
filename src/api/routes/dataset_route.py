from flask import Blueprint
from src.api.controllers.dataset_controller import DatasetController

dataset_bp = Blueprint("dataset", __name__, url_prefix="/dataset")
dataset_controller = DatasetController()

# Route untuk mengunggah dataset
dataset_bp.route("/upload", methods=["POST"]
                 )(dataset_controller.upload_dataset)

# Route untuk mengambil semua dataset
dataset_bp.route("/list", methods=["GET"])(dataset_controller.get_datasets)

# Route untuk mengambil dataset tertentu dengan paginasi
dataset_bp.route(
    "/<dataset_id>", methods=["GET"])(dataset_controller.get_dataset)

# Route untuk menghapus dataset tertentu
dataset_bp.route(
    "/<dataset_id>", methods=["DELETE"])(dataset_controller.delete_dataset)
