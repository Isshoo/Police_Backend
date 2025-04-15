from flask import Blueprint
from src.api.controllers.preprocess_controller import PreprocessController

preprocess_bp = Blueprint("preprocess", __name__, url_prefix="/dataset")
preprocess_controller = PreprocessController()


# Route untuk melakukan preprocessing dataset
preprocess_bp.route("/<raw_dataset_id>/preprocess", methods=["POST"]
                    )(preprocess_controller.preprocess_dataset)

# Route untuk membuat salinan dataset yang sudah diproses
preprocess_bp.route("/<raw_dataset_id>/preprocessed/copy", methods=["POST"]
                    )(preprocess_controller.create_preprocessed_copy)

# Route untuk mengambil dataset yang sudah diproses
preprocess_bp.route("/<raw_dataset_id>/preprocessed/list", methods=["GET"]
                    )(preprocess_controller.fetch_preprocessed_datasets)

# Route untuk mengambil dataset yang sudah diproses tertentu
preprocess_bp.route("/preprocessed/<dataset_id>", methods=["GET"]
                    )(preprocess_controller.fetch_preprocessed_dataset)

# Route untuk menghapus dataset yang sudah diproses tertentu
preprocess_bp.route("/preprocessed/<dataset_id>", methods=["DELETE"]
                    )(preprocess_controller.delete_preprocessed_dataset)

# Route untuk mengubah label manual dataset yang sudah diproses
preprocess_bp.route("/preprocessed/<dataset_id>/label", methods=["PUT"]
                    )(preprocess_controller.update_label)

# Route untuk menambah data pada dataset yang sudah diproses
preprocess_bp.route("/preprocessed/<dataset_id>/data", methods=["POST"]
                    )(preprocess_controller.add_data)

# Route untuk menghapus baris dataset yang sudah diproses
preprocess_bp.route("/preprocessed/<dataset_id>/data", methods=["DELETE"]
                    )(preprocess_controller.delete_data)
