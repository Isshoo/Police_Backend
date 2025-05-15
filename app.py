from src.api.routes import routes_bp
from flask_cors import CORS
from flask import Flask
from dotenv import load_dotenv
import logging
import os
import builtins

load_dotenv()


def disable_print(*args, **kwargs):
    """Override fungsi print untuk production"""
    pass


def create_app():
    app = Flask(__name__)
    CORS(app)

    # ===== KONFIGURASI PRODUCTION =====
    if os.environ.get('FLASK_ENV') == 'production':
        # 1. Nonaktifkan semua print()
        builtins.print = disable_print

        # 2. Setup logging yang benar
        app.logger.setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)

        # Handle Static Files untuk Production
        from whitenoise import WhiteNoise
        app.wsgi_app = WhiteNoise(app.wsgi_app, root='static/')

        # 3. (Opsional) Simpan log error ke file
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            'errors.log', maxBytes=10000, backupCount=3)
        file_handler.setLevel(logging.WARNING)
        app.logger.addHandler(file_handler)

    # Register blueprint
    app.register_blueprint(routes_bp)
    # app.register_blueprint(db_routes_bp, url_prefix="/db")

    return app


if __name__ == "__main__":
    app = create_app()

    # JANGAN gunakan debug=True di production!
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host="0.0.0.0", port=8000, debug=debug_mode)
