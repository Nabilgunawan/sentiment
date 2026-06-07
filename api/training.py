"""
api/training.py
----------------
Blueprint untuk model training admin interface.
Endpoints untuk fine-tuning IndoBERT models (sentiment, emotion, sarcasm).
"""
import os
import json
import logging
import threading
from flask import Blueprint, request, jsonify, render_template

logger = logging.getLogger(__name__)

training_bp = Blueprint("training", __name__)

# Global training state
_training_state = {
    "is_training": False,
    "model_type": None,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "train_loss": 0,
    "val_loss": 0,
    "val_accuracy": 0,
    "status": "idle",  # idle, training, completed, error
    "error": None,
    "history": None,
    "evaluation": None,
}


@training_bp.route("/training")
def training_page():
    """Render halaman admin training."""
    return render_template("training.html")


@training_bp.route("/api/training/start", methods=["POST"])
def start_training():
    """Mulai fine-tuning model."""
    global _training_state

    if _training_state["is_training"]:
        return jsonify({"error": "Training sedang berjalan. Tunggu hingga selesai."}), 400

    # Parse request
    if "file" not in request.files:
        return jsonify({"error": "Upload file dataset berlabel (CSV/XLSX)."}), 400

    file = request.files["file"]
    model_type = request.form.get("model_type", "sentiment")
    text_column = request.form.get("text_column", "text")
    label_column = request.form.get("label_column", "label")
    epochs = int(request.form.get("epochs", "5"))
    batch_size = int(request.form.get("batch_size", "16"))
    learning_rate = float(request.form.get("learning_rate", "2e-5"))

    if model_type not in ("sentiment", "emotion", "sarcasm"):
        return jsonify({"error": "model_type harus: sentiment, emotion, atau sarcasm"}), 400

    # Save uploaded file
    import config
    upload_folder = config.UPLOAD_FOLDER
    filename = f"training_{model_type}_{file.filename}"
    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    # Reset state
    _training_state = {
        "is_training": True,
        "model_type": model_type,
        "progress": 0,
        "current_epoch": 0,
        "total_epochs": epochs,
        "train_loss": 0,
        "val_loss": 0,
        "val_accuracy": 0,
        "status": "training",
        "error": None,
        "history": None,
        "evaluation": None,
    }

    # Start training in background thread
    thread = threading.Thread(
        target=_run_training,
        args=(filepath, model_type, text_column, label_column, epochs, batch_size, learning_rate),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "message": f"Training {model_type} dimulai dengan {epochs} epochs.",
        "status": "training",
    })


def _run_training(filepath, model_type, text_column, label_column, epochs, batch_size, learning_rate):
    """Background training process."""
    global _training_state

    try:
        from training.data_utils import load_training_data, validate_labels, split_data
        from training.trainer import ModelTrainer
        from training.evaluator import ModelEvaluator
        import config

        # Load data
        logger.info(f"Loading training data from {filepath}")
        texts, labels = load_training_data(filepath, text_column, label_column)

        # Validate labels
        valid_labels = {
            "sentiment": config.SENTIMENT_LABELS,
            "emotion": config.EMOTION_LABELS,
            "sarcasm": config.SARCASM_LABELS,
        }[model_type]

        is_valid, invalid = validate_labels(labels, valid_labels)
        if not is_valid:
            _training_state["status"] = "error"
            _training_state["error"] = f"Label tidak valid: {invalid}. Label yang diizinkan: {valid_labels}"
            _training_state["is_training"] = False
            return

        # Split data
        train_texts, val_texts, train_labels, val_labels = split_data(texts, labels)

        logger.info(
            f"Data split: {len(train_texts)} train, {len(val_texts)} val, "
            f"{len(set(labels))} classes"
        )

        # Training callback
        def on_progress(epoch, train_loss, val_loss, val_accuracy):
            _training_state["current_epoch"] = epoch
            _training_state["train_loss"] = round(train_loss, 4)
            _training_state["val_loss"] = round(val_loss, 4)
            _training_state["val_accuracy"] = round(val_accuracy, 4)
            _training_state["progress"] = int((epoch / epochs) * 100)

        # Train
        trainer = ModelTrainer(model_type)
        history = trainer.train(
            train_texts, train_labels,
            val_texts, val_labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            callback=on_progress,
        )

        _training_state["history"] = history

        # Evaluate
        logger.info("Evaluating trained model...")
        try:
            evaluator = ModelEvaluator(model_type)
            evaluation = evaluator.full_report(val_texts, val_labels)
            # Convert chart bytes to None for JSON serialization
            if "charts" in evaluation:
                evaluation["charts"] = {
                    k: True for k in evaluation["charts"]  # Just indicate they exist
                }
            _training_state["evaluation"] = evaluation
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")

        _training_state["status"] = "completed"
        _training_state["progress"] = 100
        logger.info(f"Training {model_type} selesai! Best accuracy: {history.get('best_val_accuracy', 0):.4f}")

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        _training_state["status"] = "error"
        _training_state["error"] = str(e)
    finally:
        _training_state["is_training"] = False


@training_bp.route("/api/training/status")
def training_status():
    """Cek status training saat ini."""
    return jsonify(_training_state)


@training_bp.route("/api/training/evaluate/<model_type>")
def get_evaluation(model_type):
    """Dapatkan metrik evaluasi model yang sudah di-train."""
    if model_type not in ("sentiment", "emotion", "sarcasm"):
        return jsonify({"error": "model_type harus: sentiment, emotion, atau sarcasm"}), 400

    try:
        from training.evaluator import ModelEvaluator
        evaluator = ModelEvaluator(model_type)

        return jsonify({
            "model_type": model_type,
            "model_loaded": evaluator._model is not None,
            "message": f"Model {model_type} siap untuk evaluasi. Upload test data untuk melihat metrik.",
        })
    except Exception as e:
        return jsonify({
            "model_type": model_type,
            "model_loaded": False,
            "message": f"Model {model_type} belum di-train atau gagal dimuat: {str(e)}",
        })


@training_bp.route("/api/training/evaluate", methods=["POST"])
def run_evaluation():
    """Jalankan evaluasi pada test dataset."""
    if "file" not in request.files:
        return jsonify({"error": "Upload file test dataset."}), 400

    file = request.files["file"]
    model_type = request.form.get("model_type", "sentiment")
    text_column = request.form.get("text_column", "text")
    label_column = request.form.get("label_column", "label")

    try:
        import config
        from training.data_utils import load_training_data
        from training.evaluator import ModelEvaluator

        # Save and load test data
        upload_folder = config.UPLOAD_FOLDER
        filepath = os.path.join(upload_folder, f"eval_{model_type}_{file.filename}")
        file.save(filepath)

        texts, labels = load_training_data(filepath, text_column, label_column)

        evaluator = ModelEvaluator(model_type)
        report = evaluator.full_report(texts, labels)

        # Convert chart bytes to base64 for frontend
        import base64
        if "charts" in report:
            for key, chart_bytes in report["charts"].items():
                if isinstance(chart_bytes, bytes):
                    report["charts"][key] = base64.b64encode(chart_bytes).decode("utf-8")

        return jsonify(report)

    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
