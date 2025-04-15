from flask import Blueprint
from src.api.routes.predict_route import predict_bp
from src.api.routes.dataset_route import dataset_bp
from src.api.routes.preprocess_route import preprocess_bp
from src.api.routes.process_route import process_bp

# Inisialisasi Blueprint utama untuk routes
routes_bp = Blueprint("routes", __name__)

# Register semua blueprint
routes_bp.register_blueprint(predict_bp)
routes_bp.register_blueprint(dataset_bp)
routes_bp.register_blueprint(preprocess_bp)
routes_bp.register_blueprint(process_bp)
