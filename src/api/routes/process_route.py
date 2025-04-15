from flask import Blueprint
from src.api.controllers.process_controller import ProcessController

process_bp = Blueprint("process", __name__)
process_controller = ProcessController()

process_bp.route("/process/split/<preprocessed_dataset_id>",
                 methods=["POST"])(process_controller.split_dataset)
process_bp.route("/process/train/<preprocessed_dataset_id>",
                 methods=["POST"])(process_controller.train_model)
process_bp.route("/process/models",
                 methods=["GET"])(process_controller.get_models)
process_bp.route("/process/model/<model_id>",
                 methods=["GET"])(process_controller.get_model)
process_bp.route("/process/model/<model_id>/name",
                 methods=["PUT"])(process_controller.edit_model_name)
process_bp.route("/process/model/<model_id>",
                 methods=["DELETE"])(process_controller.delete_model)

# Endpoint metadata tambahan & CSV pagination
process_bp.route("/process/model/<model_id>/evaluation",
                 methods=["GET"])(process_controller.get_model_evaluation)
process_bp.route("/process/model/<model_id>/parameters",
                 methods=["GET"])(process_controller.get_model_parameters)
process_bp.route("/process/model/<model_id>/tfidf-stats",
                 methods=["GET"])(process_controller.fetch_tfidf_stats)
process_bp.route("/process/model/<model_id>/neighbors",
                 methods=["GET"])(process_controller.fetch_neighbors)
