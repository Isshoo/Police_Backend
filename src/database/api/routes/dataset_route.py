from flask import Blueprint
from src.database.api.controllers.dataset_controller import DatasetController

controller = DatasetController()
dataset_db_bp = Blueprint("dataset_db", __name__)

dataset_db_bp.route("/", methods=["GET"])(controller.get_datasets)
dataset_db_bp.route("/<string:dataset_id>",
                    methods=["GET"])(controller.get_dataset_by_id)
dataset_db_bp.route("/", methods=["POST"])(controller.create_dataset)
dataset_db_bp.route("/<string:dataset_id>",
                    methods=["DELETE"])(controller.delete_dataset)
