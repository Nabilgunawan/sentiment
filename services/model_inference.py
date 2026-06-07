"""
services/model_inference.py
----------------------------
Layanan inferensi model Transformer terpusat untuk analisis sentimen, emosi, dan sarkasme.
Mengimplementasikan lazy-loading, dynamic batching, mixed precision, GPU detection, dan CPU fallback.
"""

import json
import logging
from pathlib import Path
import config
from preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# State variables for lazy loading
_SENTIMENT_MODEL = None
_SENTIMENT_TOKENIZER = None

_EMOTION_MODEL = None
_EMOTION_TOKENIZER = None

_SARCASM_MODEL = None
_SARCASM_TOKENIZER = None

# Fallback services
_SENTIMENT_FALLBACK_SVC = None
_EMOTION_FALLBACK_SVC = None
_SARCASM_FALLBACK_SVC = None

# Labels (dapat di-override dari label_mapping.json model fine-tuned)
_SENTIMENT_LABELS = list(config.SENTIMENT_LABELS)
_EMOTION_LABELS = list(config.EMOTION_LABELS)
_SARCASM_LABELS = list(config.SARCASM_LABELS)


def _load_labels_from_path(model_dir, default_labels):
    """Baca label_mapping.json dari folder model fine-tuned. Fallback ke default."""
    mapping_path = Path(model_dir) / "label_mapping.json"
    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            labels = mapping.get("labels")
            if labels and isinstance(labels, list) and len(labels) > 0:
                logger.info(f"Label dimuat dari {mapping_path}: {labels}")
                return labels
        except Exception as e:
            logger.warning(f"Gagal membaca label_mapping.json: {e}")
    return list(default_labels)


def load_sentiment_model():
    """
    Load model sentiment secara lazy-loading.
    Jika USE_TRANSFORMERS = False, menggunakan fallback lexicon secara instan.
    """
    global _SENTIMENT_MODEL, _SENTIMENT_TOKENIZER, _SENTIMENT_FALLBACK_SVC
    if not config.USE_TRANSFORMERS:
        logger.info("USE_TRANSFORMERS set to False. Menggunakan fallback lexicon-based secara instan untuk sentimen.")
        if _SENTIMENT_FALLBACK_SVC is None:
            from services.sentiment_service import SentimentService
            _SENTIMENT_FALLBACK_SVC = SentimentService()
        return None, None

    if _SENTIMENT_MODEL is not None and _SENTIMENT_TOKENIZER is not None:
        return _SENTIMENT_MODEL, _SENTIMENT_TOKENIZER

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = config.resolve_device()
        finetuned_path = Path(config.SENTIMENT_FINETUNED_PATH)

        # Cek model fine-tuned di lokal
        if finetuned_path.exists() and any((finetuned_path / f).exists() for f in ["pytorch_model.bin", "model.safetensors", "config.json"]):
            global _SENTIMENT_LABELS
            logger.info(f"Memuat model sentimen fine-tuned dari: {finetuned_path}")
            _SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(str(finetuned_path))
            _SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained(str(finetuned_path))
            _SENTIMENT_LABELS = _load_labels_from_path(finetuned_path, config.SENTIMENT_LABELS)
        else:
            # Cegah load raw model MLM tanpa training
            raw_models = ["indobenchmark/indobert-base-p1", "indobenchmark/indobert-base-p2", "indobenchmark/indobert-large-p1", "bert-base-multilingual-cased"]
            if config.SENTIMENT_MODEL_NAME in raw_models:
                logger.info(f"Model pre-trained '{config.SENTIMENT_MODEL_NAME}' adalah raw model tanpa head klasifikasi terlatih. Menggunakan fallback lexicon.")
                if _SENTIMENT_FALLBACK_SVC is None:
                    from services.sentiment_service import SentimentService
                    _SENTIMENT_FALLBACK_SVC = SentimentService()
                return None, None

            logger.info(f"Memuat model sentimen pre-trained: {config.SENTIMENT_MODEL_NAME}")
            _SENTIMENT_TOKENIZER = AutoTokenizer.from_pretrained(config.SENTIMENT_MODEL_NAME)
            _SENTIMENT_MODEL = AutoModelForSequenceClassification.from_pretrained(config.SENTIMENT_MODEL_NAME, num_labels=len(config.SENTIMENT_LABELS))

        _SENTIMENT_MODEL = _SENTIMENT_MODEL.to(device)
        _SENTIMENT_MODEL.eval()
        logger.info(f"Model sentimen berhasil dimuat pada device: {device}")
    except Exception as e:
        logger.error(f"Gagal memuat model sentimen: {e}. Menggunakan fallback lexicon-based.")
        if _SENTIMENT_FALLBACK_SVC is None:
            from services.sentiment_service import SentimentService
            _SENTIMENT_FALLBACK_SVC = SentimentService()

    return _SENTIMENT_MODEL, _SENTIMENT_TOKENIZER


def load_emotion_model():
    """
    Load model emotion secara lazy-loading.
    Jika USE_TRANSFORMERS = False, menggunakan fallback lexicon secara instan.
    """
    global _EMOTION_MODEL, _EMOTION_TOKENIZER, _EMOTION_FALLBACK_SVC
    if not config.USE_TRANSFORMERS:
        logger.info("USE_TRANSFORMERS set to False. Menggunakan fallback lexicon-based secara instan untuk emosi.")
        if _EMOTION_FALLBACK_SVC is None:
            from services.emotion_service import EmotionService
            _EMOTION_FALLBACK_SVC = EmotionService()
        return None, None

    if _EMOTION_MODEL is not None and _EMOTION_TOKENIZER is not None:
        return _EMOTION_MODEL, _EMOTION_TOKENIZER

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = config.resolve_device()
        finetuned_path = Path(config.EMOTION_FINETUNED_PATH)

        # Cek model fine-tuned di lokal
        if finetuned_path.exists() and any((finetuned_path / f).exists() for f in ["pytorch_model.bin", "model.safetensors", "config.json"]):
            global _EMOTION_LABELS
            logger.info(f"Memuat model emosi fine-tuned dari: {finetuned_path}")
            _EMOTION_TOKENIZER = AutoTokenizer.from_pretrained(str(finetuned_path))
            _EMOTION_MODEL = AutoModelForSequenceClassification.from_pretrained(str(finetuned_path))
            _EMOTION_LABELS = _load_labels_from_path(finetuned_path, config.EMOTION_LABELS)
        else:
            raw_models = ["indobenchmark/indobert-base-p1", "indobenchmark/indobert-base-p2", "indobenchmark/indobert-large-p1", "bert-base-multilingual-cased"]
            if config.EMOTION_MODEL_NAME in raw_models:
                logger.info(f"Model pre-trained '{config.EMOTION_MODEL_NAME}' adalah raw model tanpa head klasifikasi terlatih. Menggunakan fallback Plutchik.")
                if _EMOTION_FALLBACK_SVC is None:
                    from services.emotion_service import EmotionService
                    _EMOTION_FALLBACK_SVC = EmotionService()
                return None, None

            logger.info(f"Memuat model emosi pre-trained: {config.EMOTION_MODEL_NAME}")
            _EMOTION_TOKENIZER = AutoTokenizer.from_pretrained(config.EMOTION_MODEL_NAME)
            _EMOTION_MODEL = AutoModelForSequenceClassification.from_pretrained(config.EMOTION_MODEL_NAME, num_labels=len(config.EMOTION_LABELS))

        _EMOTION_MODEL = _EMOTION_MODEL.to(device)
        _EMOTION_MODEL.eval()
        logger.info(f"Model emosi berhasil dimuat pada device: {device}")
    except Exception as e:
        logger.error(f"Gagal memuat model emosi: {e}. Menggunakan fallback Plutchik.")
        if _EMOTION_FALLBACK_SVC is None:
            from services.emotion_service import EmotionService
            _EMOTION_FALLBACK_SVC = EmotionService()

    return _EMOTION_MODEL, _EMOTION_TOKENIZER


def load_sarcasm_model():
    """
    Load model sarcasm secara lazy-loading.
    Jika USE_TRANSFORMERS = False, menggunakan fallback rule-based secara instan.
    """
    global _SARCASM_MODEL, _SARCASM_TOKENIZER, _SARCASM_FALLBACK_SVC
    if not config.USE_TRANSFORMERS:
        logger.info("USE_TRANSFORMERS set to False. Menggunakan fallback rule-based secara instan untuk sarkasme.")
        if _SARCASM_FALLBACK_SVC is None:
            from services.sarcasm_service import SarcasmService
            _SARCASM_FALLBACK_SVC = SarcasmService()
        return None, None

    if _SARCASM_MODEL is not None and _SARCASM_TOKENIZER is not None:
        return _SARCASM_MODEL, _SARCASM_TOKENIZER

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = config.resolve_device()
        finetuned_path = Path(config.SARCASM_FINETUNED_PATH)

        # Cek model fine-tuned di lokal
        if finetuned_path.exists() and any((finetuned_path / f).exists() for f in ["pytorch_model.bin", "model.safetensors", "config.json"]):
            global _SARCASM_LABELS
            logger.info(f"Memuat model sarkasme fine-tuned dari: {finetuned_path}")
            _SARCASM_TOKENIZER = AutoTokenizer.from_pretrained(str(finetuned_path))
            _SARCASM_MODEL = AutoModelForSequenceClassification.from_pretrained(str(finetuned_path))
            _SARCASM_LABELS = _load_labels_from_path(finetuned_path, config.SARCASM_LABELS)
        else:
            raw_models = ["indobenchmark/indobert-base-p1", "indobenchmark/indobert-base-p2", "indobenchmark/indobert-large-p1", "bert-base-multilingual-cased"]
            if config.SARCASM_MODEL_NAME in raw_models:
                logger.info(f"Model pre-trained '{config.SARCASM_MODEL_NAME}' adalah raw model tanpa head klasifikasi terlatih. Menggunakan fallback rule-based.")
                if _SARCASM_FALLBACK_SVC is None:
                    from services.sarcasm_service import SarcasmService
                    _SARCASM_FALLBACK_SVC = SarcasmService()
                return None, None

            logger.info(f"Memuat model sarkasme pre-trained: {config.SARCASM_MODEL_NAME}")
            _SARCASM_TOKENIZER = AutoTokenizer.from_pretrained(config.SARCASM_MODEL_NAME)
            _SARCASM_MODEL = AutoModelForSequenceClassification.from_pretrained(config.SARCASM_MODEL_NAME, num_labels=len(config.SARCASM_LABELS))

        _SARCASM_MODEL = _SARCASM_MODEL.to(device)
        _SARCASM_MODEL.eval()
        logger.info(f"Model sarkasme berhasil dimuat pada device: {device}")
    except Exception as e:
        logger.error(f"Gagal memuat model sarkasme: {e}. Menggunakan fallback rule-based.")
        if _SARCASM_FALLBACK_SVC is None:
            from services.sarcasm_service import SarcasmService
            _SARCASM_FALLBACK_SVC = SarcasmService()

    return _SARCASM_MODEL, _SARCASM_TOKENIZER


def predict_sentiment(text: str) -> dict:
    """Prediksi sentimen untuk satu teks."""
    if not text or not isinstance(text, str) or not text.strip():
        return _empty_result_for_task("sentiment")

    model, tokenizer = load_sentiment_model()
    if model is None or tokenizer is None:
        return _SENTIMENT_FALLBACK_SVC.predict(text)

    try:
        import torch
        device = config.resolve_device()
        cleaned = preprocess_text(text)
        if not cleaned.strip():
            return _empty_result_for_task("sentiment")

        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        use_amp = (device == "cuda")
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]
        predicted_idx = int(probs_np.argmax())
        label = _SENTIMENT_LABELS[predicted_idx]
        confidence = float(probs_np[predicted_idx])

        result = {
            "sentiment": label,
            "confidence": round(confidence, 4),
        }
        for i, lbl in enumerate(_SENTIMENT_LABELS):
            result[f"prob_{lbl.lower()}"] = round(float(probs_np[i]), 4)
        return result
    except Exception as e:
        logger.error(f"Inference error pada predict_sentiment: {e}. Menggunakan fallback.")
        if _SENTIMENT_FALLBACK_SVC is None:
            from services.sentiment_service import SentimentService
            _SENTIMENT_FALLBACK_SVC = SentimentService()
        return _SENTIMENT_FALLBACK_SVC.predict(text)


def predict_emotion(text: str) -> dict:
    """Prediksi emosi untuk satu teks."""
    if not text or not isinstance(text, str) or not text.strip():
        return _empty_result_for_task("emotion")

    model, tokenizer = load_emotion_model()
    if model is None or tokenizer is None:
        return _EMOTION_FALLBACK_SVC.predict(text)

    try:
        import torch
        device = config.resolve_device()
        cleaned = preprocess_text(text)
        if not cleaned.strip():
            return _empty_result_for_task("emotion")

        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        use_amp = (device == "cuda")
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]
        predicted_idx = int(probs_np.argmax())
        label = _EMOTION_LABELS[predicted_idx]
        confidence = float(probs_np[predicted_idx])

        scores = {
            _EMOTION_LABELS[i]: round(float(probs_np[i]), 4)
            for i in range(len(_EMOTION_LABELS))
        }

        return {
            "emotion": label,
            "confidence": round(confidence, 4),
            "scores": scores,
        }
    except Exception as e:
        logger.error(f"Inference error pada predict_emotion: {e}. Menggunakan fallback.")
        if _EMOTION_FALLBACK_SVC is None:
            from services.emotion_service import EmotionService
            _EMOTION_FALLBACK_SVC = EmotionService()
        return _EMOTION_FALLBACK_SVC.predict(text)


def predict_sarcasm(text: str) -> dict:
    """Prediksi sarkasme untuk satu teks."""
    if not text or not isinstance(text, str) or not text.strip():
        return _empty_result_for_task("sarcasm")

    model, tokenizer = load_sarcasm_model()
    if model is None or tokenizer is None:
        return _SARCASM_FALLBACK_SVC.predict(text)

    try:
        import torch
        device = config.resolve_device()
        cleaned = preprocess_text(text)
        if not cleaned.strip():
            return _empty_result_for_task("sarcasm")

        inputs = tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        use_amp = (device == "cuda")
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]
        predicted_idx = int(probs_np.argmax())
        label = _SARCASM_LABELS[predicted_idx]
        confidence = float(probs_np[predicted_idx])

        return {
            "sarcasm": label == "Sarcasm",
            "confidence": round(confidence, 4),
        }
    except Exception as e:
        logger.error(f"Inference error pada predict_sarcasm: {e}. Menggunakan fallback.")
        if _SARCASM_FALLBACK_SVC is None:
            from services.sarcasm_service import SarcasmService
            _SARCASM_FALLBACK_SVC = SarcasmService()
        return _SARCASM_FALLBACK_SVC.predict(text)


def batch_predict(texts: list[str], task: str = "sentiment") -> list[dict]:
    """Prediksi batch dengan dynamic batching dan mixed precision."""
    if not texts:
        return []

    if task == "sentiment":
        model, tokenizer = load_sentiment_model()
        fallback_svc = _SENTIMENT_FALLBACK_SVC
        labels = _SENTIMENT_LABELS
    elif task == "emotion":
        model, tokenizer = load_emotion_model()
        fallback_svc = _EMOTION_FALLBACK_SVC
        labels = _EMOTION_LABELS
    elif task == "sarcasm":
        model, tokenizer = load_sarcasm_model()
        fallback_svc = _SARCASM_FALLBACK_SVC
        labels = _SARCASM_LABELS
    else:
        raise ValueError(f"Unknown task: {task}")

    if model is None or tokenizer is None:
        if fallback_svc is None:
            if task == "sentiment":
                from services.sentiment_service import SentimentService
                fallback_svc = SentimentService()
            elif task == "emotion":
                from services.emotion_service import EmotionService
                fallback_svc = EmotionService()
            elif task == "sarcasm":
                from services.sarcasm_service import SarcasmService
                fallback_svc = SarcasmService()
        return fallback_svc.predict_batch(texts)

    results = [None] * len(texts)
    valid_indices = []
    valid_texts = []

    for idx, t in enumerate(texts):
        if t and isinstance(t, str) and t.strip():
            cleaned = preprocess_text(t)
            if cleaned.strip():
                valid_indices.append(idx)
                valid_texts.append(cleaned)
            else:
                results[idx] = _empty_result_for_task(task)
        else:
            results[idx] = _empty_result_for_task(task)

    if not valid_texts:
        return [r if r is not None else _empty_result_for_task(task) for r in results]

    try:
        import torch
        device = config.resolve_device()
        batch_size = config.BATCH_SIZE
        use_amp = (device == "cuda")

        for i in range(0, len(valid_texts), batch_size):
            chunk_texts = valid_texts[i : i + batch_size]
            chunk_indices = valid_indices[i : i + batch_size]

            inputs = tokenizer(
                chunk_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.MAX_SEQ_LENGTH,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)

                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

            probs_np = probs.cpu().numpy()

            for batch_idx, orig_idx in enumerate(chunk_indices):
                p = probs_np[batch_idx]
                predicted_idx = int(p.argmax())
                label = labels[predicted_idx]
                confidence = float(p[predicted_idx])

                if task == "sentiment":
                    result = {
                        "sentiment": label,
                        "confidence": round(confidence, 4),
                    }
                    for i, lbl in enumerate(labels):
                        result[f"prob_{lbl.lower()}"] = round(float(p[i]), 4)
                    results[orig_idx] = result
                elif task == "emotion":
                    scores = {labels[k]: round(float(p[k]), 4) for k in range(len(labels))}
                    results[orig_idx] = {
                        "emotion": label,
                        "confidence": round(confidence, 4),
                        "scores": scores,
                    }
                elif task == "sarcasm":
                    results[orig_idx] = {
                        "sarcasm": label == "Sarcasm",
                        "confidence": round(confidence, 4),
                    }
    except Exception as e:
        logger.error(f"Error pada batch_predict task {task}: {e}. Menggunakan fallback individual.")
        for idx in valid_indices:
            if results[idx] is None:
                try:
                    results[idx] = fallback_svc.predict(texts[idx])
                except Exception:
                    results[idx] = _empty_result_for_task(task)

    return [r if r is not None else _empty_result_for_task(task) for r in results]


def _empty_result_for_task(task: str) -> dict:
    if task == "sentiment":
        result = {
            "sentiment": "Neutral",
            "confidence": 0.0,
        }
        for lbl in _SENTIMENT_LABELS:
            result[f"prob_{lbl.lower()}"] = 1.0 if lbl.lower() == "neutral" else 0.0
        return result
    elif task == "emotion":
        scores = {e: 0.0 for e in _EMOTION_LABELS}
        if "Netral" in scores:
            scores["Netral"] = 1.0
        return {
            "emotion": "Netral",
            "confidence": 0.0,
            "scores": scores,
        }
    elif task == "sarcasm":
        return {
            "sarcasm": False,
            "confidence": 0.0,
        }
