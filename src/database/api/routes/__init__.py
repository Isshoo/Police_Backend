from flask import Blueprint
from src.database.api.routes.dataset_route import dataset_db_bp

db_routes_bp = Blueprint("db_routes", __name__)
db_routes_bp.register_blueprint(dataset_db_bp, url_prefix="/datasets")
