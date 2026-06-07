"""
api/model_routes.py
--------------------
Blueprint untuk endpoints manajemen model, inferensi, status, dan metrik.
Endpoints:
- POST /train/sentiment
- POST /train/emotion
- POST /train/sarcasm
- POST /predict
- GET /models/status
- GET /models/metrics
"""

import os
import json
import logging
import zipfile
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
import config

logger = logging.getLogger(__name__)

model_routes_bp = Blueprint("model_routes", __name__)


def safe_extract(zip_path, target_dir):
    """Mengekstrak file ZIP dengan aman untuk mencegah serangan path traversal."""
    target_path = os.path.abspath(target_dir)
    os.makedirs(target_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            member_path = os.path.abspath(os.path.join(target_path, member.filename))
            if not member_path.startswith(target_path + os.sep) and member_path != target_path:
                raise Exception(f"Deteksi ancaman path traversal dalam file ZIP: {member.filename}")
        zip_ref.extractall(target_path)


def flatten_model_dir(target_dir):
    """
    Jika ZIP memiliki subfolder (misal sentiment_model/config.json), 
    pindahkan semua file dari subfolder ke root target_dir.
    """
    import shutil

    target_path = os.path.abspath(target_dir)
    model_files = {"config.json", "pytorch_model.bin", "model.safetensors", "tokenizer_config.json",
                   "vocab.txt", "special_tokens_map.json", "training_args.bin", "metrics.json"}

    existing = {f for f in model_files if os.path.isfile(os.path.join(target_path, f))}
    if existing:
        return

    for entry in os.listdir(target_path):
        sub = os.path.join(target_path, entry)
        if not os.path.isdir(sub):
            continue
        sub_files = os.listdir(sub)
        has_model = any(f in sub_files for f in model_files)
        if has_model:
            logger.info(f"Flatten: memindahkan {len(sub_files)} file dari '{entry}' ke '{target_dir}'")
            for fname in sub_files:
                src = os.path.join(sub, fname)
                dst = os.path.join(target_path, fname)
                if os.path.isdir(src):
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
                else:
                    if os.path.exists(dst):
                        os.remove(dst)
                    shutil.move(src, dst)
            shutil.rmtree(sub)


def reload_model_in_inference(task):
    """Mereset model loaded status di model_inference agar memicu reload otomatis."""
    try:
        import services.model_inference as mi
        if task == "sentiment":
            mi._SENTIMENT_MODEL = None
            mi._SENTIMENT_TOKENIZER = None
            logger.info("Status model sentiment di-reset untuk hot reload.")
        elif task == "emotion":
            mi._EMOTION_MODEL = None
            mi._EMOTION_TOKENIZER = None
            logger.info("Status model emotion di-reset untuk hot reload.")
        elif task == "sarcasm":
            mi._SARCASM_MODEL = None
            mi._SARCASM_TOKENIZER = None
            logger.info("Status model sarcasm di-reset untuk hot reload.")
    except Exception as e:
        logger.error(f"Gagal mereset status model {task} untuk hot reload: {e}")


@model_routes_bp.route("/train/<task>", methods=["POST"])
def upload_trained_model(task):
    """
    Mengunggah model hasil training dari Colab/Kaggle dalam bentuk ZIP.
    Menghapus model lama, mengekstrak model baru, dan melakukan hot reload.
    """
    if task not in ("sentiment", "emotion", "sarcasm"):
        return jsonify({"error": "Task tidak didukung. Pilih: sentiment, emotion, sarcasm"}), 400

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file ZIP yang diunggah."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    if not file.filename.endswith(".zip"):
        return jsonify({"error": "Format file tidak didukung. Harus berformat .zip"}), 400

    # Tentukan direktori tujuan
    if task == "sentiment":
        target_dir = config.SENTIMENT_FINETUNED_PATH
    elif task == "emotion":
        target_dir = config.EMOTION_FINETUNED_PATH
    else:
        target_dir = config.SARCASM_FINETUNED_PATH

    # Simpan file ZIP sementara
    temp_zip_path = os.path.join(config.UPLOAD_FOLDER, f"temp_model_{task}_{uuid_hex()}.zip")
    try:
        file.save(temp_zip_path)
        logger.info(f"Mengekstrak model {task} baru ke: {target_dir}")

        # Bersihkan direktori lama jika ada
        import shutil
        if os.path.exists(target_dir):
            # Hapus isi folder tapi pertahankan foldernya
            for item in os.listdir(target_dir):
                item_path = os.path.join(target_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

        # Ekstrak ZIP aman
        safe_extract(temp_zip_path, target_dir)

        # Flatten jika ZIP mengandung subfolder
        flatten_model_dir(target_dir)

        # Triger hot reload
        reload_model_in_inference(task)

        # Bersihkan file ZIP temp
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

        return jsonify({
            "message": f"Model {task} berhasil diunggah, diekstrak, dan di-hot-reload.",
            "target_directory": str(target_dir),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Gagal memproses unggahan model {task}: {e}", exc_info=True)
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)
        return jsonify({"error": f"Gagal memproses unggahan model: {str(e)}"}), 500


@model_routes_bp.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint inferensi terpusat untuk klasifikasi sentiment, emotion, dan sarcasm.
    Menerima body JSON:
    {
       "text": "teks tunggal untuk diproses",
       "texts": ["list", "teks", "banyak"]  # Opsional untuk batch prediction
    }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    texts = data.get("texts")

    import services.model_inference as mi

    if text is not None:
        if not isinstance(text, str):
            return jsonify({"error": "Input 'text' harus berupa string."}), 400
        
        # Single Prediction
        sentiment_res = mi.predict_sentiment(text)
        emotion_res = mi.predict_emotion(text)
        sarcasm_res = mi.predict_sarcasm(text)

        return jsonify({
            "text": text,
            "sentiment": sentiment_res,
            "emotion": emotion_res,
            "sarcasm": sarcasm_res
        })

    elif texts is not None:
        if not isinstance(texts, list):
            return jsonify({"error": "Input 'texts' harus berupa list string."}), 400
        
        # Batch Prediction
        texts = [str(t) for t in texts]
        sentiment_results = mi.batch_predict(texts, task="sentiment")
        emotion_results = mi.batch_predict(texts, task="emotion")
        sarcasm_results = mi.batch_predict(texts, task="sarcasm")

        predictions = []
        for i, t in enumerate(texts):
            predictions.append({
                "text": t,
                "sentiment": sentiment_results[i],
                "emotion": emotion_results[i],
                "sarcasm": sarcasm_results[i]
            })

        return jsonify({
            "total": len(texts),
            "predictions": predictions
        })

    else:
        return jsonify({"error": "Harap sertakan parameter 'text' atau 'texts' pada body request."}), 400


@model_routes_bp.route("/models/status", methods=["GET"])
def models_status():
    """Mengembalikan status loading dan preferensi hardware dari ketiga model."""
    import services.model_inference as mi
    device = config.resolve_device()

    status = {
        "device": device,
        "use_transformers": config.USE_TRANSFORMERS,
        "sentiment": {
            "loaded": mi._SENTIMENT_MODEL is not None,
            "mode": "Transformer" if (mi._SENTIMENT_MODEL is not None) else "Lexicon Fallback",
            "model_name_or_path": config.SENTIMENT_MODEL_NAME if mi._SENTIMENT_MODEL is None else str(config.SENTIMENT_FINETUNED_PATH)
        },
        "emotion": {
            "loaded": mi._EMOTION_MODEL is not None,
            "mode": "Transformer" if (mi._EMOTION_MODEL is not None) else "Lexicon Fallback",
            "model_name_or_path": config.EMOTION_MODEL_NAME if mi._EMOTION_MODEL is None else str(config.EMOTION_FINETUNED_PATH)
        },
        "sarcasm": {
            "loaded": mi._SARCASM_MODEL is not None,
            "mode": "Transformer" if (mi._SARCASM_MODEL is not None) else "Lexicon Fallback",
            "model_name_or_path": config.SARCASM_MODEL_NAME if mi._SARCASM_MODEL is None else str(config.SARCASM_FINETUNED_PATH)
        }
    }
    return jsonify(status)


@model_routes_bp.route("/models/metrics", methods=["GET"])
def models_metrics():
    """Membaca dan mengembalikan file metrics.json dari masing-masing model jika tersedia."""
    metrics = {}
    tasks = {
        "sentiment": [
            Path(config.SENTIMENT_FINETUNED_PATH) / "metrics.json",
            Path(current_app.root_path) / "training" / "outputs" / "sentiment_metrics.json"
        ],
        "emotion": [
            Path(config.EMOTION_FINETUNED_PATH) / "metrics.json",
            Path(current_app.root_path) / "training" / "outputs" / "emotion_metrics.json"
        ],
        "sarcasm": [
            Path(config.SARCASM_FINETUNED_PATH) / "metrics.json",
            Path(current_app.root_path) / "training" / "outputs" / "sarcasm_metrics.json"
        ]
    }

    for task_name, paths in tasks.items():
        metrics[task_name] = None
        for path in paths:
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        metrics[task_name] = json.load(f)
                    break  # Berhenti jika f1 / metrics berhasil dimuat
                except Exception as e:
                    logger.warning(f"Gagal membaca file metrik di {path}: {e}")

    return jsonify(metrics)


def uuid_hex():
    import uuid
    return uuid.uuid4().hex
