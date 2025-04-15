from flask import Blueprint
from src.api.controllers.predict_controller import PredictController

predict_bp = Blueprint("predict", __name__)
predict_controller = PredictController()

predict_bp.route("/predict", methods=["POST"])(predict_controller.predict)
predict_bp.route(
    "/predict/csv", methods=["POST"])(predict_controller.predict_from_csv)
