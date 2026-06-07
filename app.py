"""
app.py
-------
Entry point utama Flask application.
Platform Analisis Sentimen Media Sosial Indonesia.

Arsitektur: Flask Blueprints
Engine: IndoBERT Transformer (sentiment, emotion, sarcasm)
Topic: BERTopic
Summary: DeepSeek AI (optional)
"""
import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, render_template
import config

logger = logging.getLogger(__name__)


def create_app():
    """Flask application factory."""
    app = Flask(__name__)

    # Configuration
    app.secret_key = config.SECRET_KEY
    app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Register Blueprints
    from api.upload import upload_bp
    from api.export import export_bp
    from api.training import training_bp
    from api.dashboard import dashboard_bp
    from api.model_routes import model_routes_bp

    app.register_blueprint(upload_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(model_routes_bp)

    # Main page route
    @app.route("/")
    def index():
        return render_template("index.html")

    logger.info("=" * 60)
    logger.info("  Analisis Sentimen Media Sosial Indonesia")
    logger.info("  Engine: IndoBERT Transformer")
    logger.info(f"  Device: {config.resolve_device()}")
    logger.info(f"  DeepSeek Summary: {'Aktif' if config.DEEPSEEK_API_KEY else 'Tidak aktif'}")
    logger.info("=" * 60)

    return app


# Create app instance
app = create_app()

if __name__ == "__main__":
    app.run(
        debug=config.FLASK_DEBUG,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
    )
