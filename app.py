from flask import Flask
from flask_cors import CORS

# Import Blueprint API default (src/api)
from src.api.routes import routes_bp

# Import Blueprint dari API DB alternatif (src/database/api)
from src.database.api.routes import db_routes_bp


def create_app():
    app = Flask(__name__)
    CORS(app)

    # Register blueprint lama (file-based API)
    app.register_blueprint(routes_bp)

    # Register blueprint baru (DB-based API)
    app.register_blueprint(db_routes_bp, url_prefix="/db")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)
