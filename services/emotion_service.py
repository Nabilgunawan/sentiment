"""
services/emotion_service.py
----------------------------
Layanan deteksi emosi Bahasa Indonesia menggunakan model Transformer (IndoBERT).
Mendukung 7 kelas emosi berbasis model Plutchik yang disesuaikan.

Kelas emosi:
- Marah, Senang, Sedih, Takut, Jijik, Terkejut, Netral

Fitur utama:
- Load model fine-tuned dari path lokal, fallback ke pre-trained
- Jika semua model gagal, gunakan fallback lexicon Plutchik
- Prediksi tunggal dan batch dengan dynamic batching
- Softmax probability distribution untuk setiap emosi
"""

import logging
from typing import Optional
from collections import defaultdict
from pathlib import Path

import config
from preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# ── Lexicon Emosi Plutchik (7 Emosi) ────────────────────────────────────────

EMOTION_LEXICON: dict[str, set[str]] = {
    "Marah": {
        "marah", "kesal", "geram", "murka", "jengkel", "benci", "dongkol",
        "sewot", "gondok", "nyebelin", "brengsek", "bangsat", "sialan",
    },
    "Senang": {
        "senang", "bahagia", "gembira", "suka", "ceria", "girang",
        "bersyukur", "alhamdulillah", "terima kasih", "mantap", "keren",
        "bagus", "puas", "lega", "seru",
    },
    "Sedih": {
        "sedih", "pilu", "menangis", "duka", "kecewa", "patah hati",
        "murung", "miris", "menderita", "hancur", "kehilangan", "rindu",
    },
    "Takut": {
        "takut", "khawatir", "cemas", "panik", "gelisah", "ngeri",
        "trauma", "gugup", "tegang", "pesimis", "putus asa",
    },
    "Jijik": {
        "jijik", "muak", "mual", "memuakkan", "kotor", "jorok", "busuk",
        "vulgar", "sampah", "hina",
    },
    "Terkejut": {
        "kaget", "terkejut", "shock", "wow", "astaga", "gila", "gokil",
        "luar biasa", "fantastis", "dahsyat", "amazing",
    },
}

# Bobot emosi untuk normalisasi skor lexicon
_EMOTION_WEIGHTS: dict[str, float] = {
    "Marah": 1.2,
    "Senang": 1.0,
    "Sedih": 1.0,
    "Takut": 1.1,
    "Jijik": 1.1,
    "Terkejut": 0.9,
    "Netral": 0.5,
}


class EmotionService:
    """
    Layanan deteksi emosi berbasis Transformer dengan fallback lexicon Plutchik.

    Alur inisialisasi:
    1. Coba load model fine-tuned dari EMOTION_FINETUNED_PATH
    2. Jika tidak ada, load pre-trained dari EMOTION_MODEL_NAME
    3. Jika semua gagal, gunakan fallback lexicon-based

    Attributes:
        model: Model Transformer untuk klasifikasi emosi
        tokenizer: Tokenizer yang cocok dengan model
        device: Device untuk inference (cuda/cpu)
        labels: Daftar label emosi
    """

    def __init__(self) -> None:
        """Inisialisasi EmotionService dengan strategi loading bertingkat."""
        self.model = None
        self.tokenizer = None
        self.device: str = config.resolve_device()
        self.labels: list[str] = config.EMOTION_LABELS
        self._use_fallback: bool = False

        # Bangun index terbalik untuk fallback lexicon: kata → list emosi
        self._word_index: dict[str, list[str]] = defaultdict(list)
        for emotion, words in EMOTION_LEXICON.items():
            for word in words:
                self._word_index[word.lower()].append(emotion)

        self._load_model()

    # ── Model Loading ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Muat model Transformer dengan strategi bertingkat:
        fine-tuned → pre-trained → fallback lexicon.
        """
        if not config.USE_TRANSFORMERS:
            logger.info("USE_TRANSFORMERS set to False. Menggunakan fallback lexicon-based secara instan untuk emosi.")
            self._use_fallback = True
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            logger.warning(
                "Library transformers/torch tidak terinstall. "
                "Menggunakan fallback lexicon-based untuk emosi."
            )
            self._use_fallback = True
            return

        # Strategi 1: Load model fine-tuned
        finetuned_path = Path(config.EMOTION_FINETUNED_PATH)
        if finetuned_path.exists() and self._is_valid_model_dir(finetuned_path):
            try:
                logger.info(
                    f"Memuat model emosi fine-tuned dari: {finetuned_path}"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(str(finetuned_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(finetuned_path)
                )
                self._move_model_to_device()
                logger.info("Model emosi fine-tuned berhasil dimuat.")
                return
            except Exception as e:
                logger.warning(
                    f"Gagal memuat model emosi fine-tuned: {e}. "
                    "Mencoba model pre-trained..."
                )

        # Cek jika model pre-trained adalah raw language model tanpa classification head terlatih.
        # Menggunakan fallback lexicon-based jika model-nya raw, untuk mencegah prediksi acak.
        raw_models = ["indobenchmark/indobert-base-p1", "indobenchmark/indobert-base-p2", "indobenchmark/indobert-large-p1", "bert-base-multilingual-cased"]
        if config.EMOTION_MODEL_NAME in raw_models:
            logger.info(
                f"Model pre-trained '{config.EMOTION_MODEL_NAME}' adalah raw model tanpa head klasifikasi terlatih. "
                "Menggunakan fallback lexicon-based untuk akurasi dasar (sebelum model di-training)."
            )
            self._use_fallback = True
            logger.info("EmotionService diinisialisasi dengan fallback lexicon Plutchik.")
            return

        # Strategi 2: Load model pre-trained dari HuggingFace
        try:
            model_name = config.EMOTION_MODEL_NAME
            logger.info(f"Memuat model emosi pre-trained: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(self.labels)
            )
            self._move_model_to_device()
            logger.info("Model emosi pre-trained berhasil dimuat.")
            return
        except Exception as e:
            logger.warning(
                f"Gagal memuat model emosi pre-trained: {e}. "
                "Menggunakan fallback lexicon-based."
            )

        # Strategi 3: Fallback ke lexicon
        self._use_fallback = True
        logger.info("EmotionService diinisialisasi dengan fallback lexicon Plutchik.")

    def _is_valid_model_dir(self, path: Path) -> bool:
        """Periksa apakah direktori berisi file model yang valid."""
        required_files = ["config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]
        has_config = any((path / f).exists() for f in required_files)
        has_model = any((path / f).exists() for f in model_files)
        return has_config and has_model

    def _move_model_to_device(self) -> None:
        """Pindahkan model ke device yang sesuai dan set ke mode evaluasi."""
        if self.model is not None:
            import torch  # noqa: F811
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model emosi dipindahkan ke device: {self.device}")

    # ── Prediksi Tunggal ─────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Prediksi emosi untuk satu teks.

        Args:
            text: Teks mentah yang akan dianalisis

        Returns:
            Dictionary berisi:
                - emotion (str): Label emosi dominan
                - confidence (float): Skor kepercayaan (0.0-1.0)
                - scores (dict): Skor probabilitas per emosi
        """
        if not text or not isinstance(text, str) or not text.strip():
            return self._empty_result()

        if self._use_fallback:
            return self._fallback_predict(text)

        try:
            return self._model_predict(text)
        except Exception as e:
            logger.error(f"Error pada prediksi model emosi: {e}")
            return self._fallback_predict(text)

    def _model_predict(self, text: str) -> dict:
        """Prediksi menggunakan model Transformer."""
        import torch

        cleaned = preprocess_text(text)
        if not cleaned.strip():
            return self._empty_result()

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.MAX_SEQ_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]
        predicted_idx = int(probs_np.argmax())
        label = self.labels[predicted_idx]
        confidence = float(probs_np[predicted_idx])

        scores = {
            self.labels[i]: round(float(probs_np[i]), 4)
            for i in range(len(self.labels))
        }

        return {
            "emotion": label,
            "confidence": round(confidence, 4),
            "scores": scores,
        }

    # ── Prediksi Batch ───────────────────────────────────────────────────

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Prediksi emosi untuk batch teks dengan dynamic batching.

        Args:
            texts: List teks mentah yang akan dianalisis

        Returns:
            List dictionary hasil prediksi untuk setiap teks
        """
        if not texts:
            return []

        if self._use_fallback:
            return [self._fallback_predict(t) for t in texts]

        results: list[dict] = []
        batch_size = config.BATCH_SIZE

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_results = self._predict_batch_chunk(batch)
            results.extend(batch_results)

        return results

    def _predict_batch_chunk(self, texts: list[str]) -> list[dict]:
        """Prediksi satu chunk batch menggunakan model Transformer."""
        import torch

        cleaned_texts = [
            preprocess_text(t) if t and isinstance(t, str) else "" for t in texts
        ]

        results: list[Optional[dict]] = [None] * len(texts)
        valid_indices: list[int] = []
        valid_texts: list[str] = []

        for idx, ct in enumerate(cleaned_texts):
            if ct.strip():
                valid_indices.append(idx)
                valid_texts.append(ct)
            else:
                results[idx] = self._empty_result()

        if not valid_texts:
            return [r if r is not None else self._empty_result() for r in results]

        try:
            inputs = self.tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.MAX_SEQ_LENGTH,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

            probs_np = probs.cpu().numpy()

            for batch_idx, orig_idx in enumerate(valid_indices):
                p = probs_np[batch_idx]
                predicted_idx = int(p.argmax())
                label = self.labels[predicted_idx]
                confidence = float(p[predicted_idx])

                scores = {
                    self.labels[i]: round(float(p[i]), 4)
                    for i in range(len(self.labels))
                }

                results[orig_idx] = {
                    "emotion": label,
                    "confidence": round(confidence, 4),
                    "scores": scores,
                }
        except Exception as e:
            logger.error(f"Error pada batch prediksi emosi: {e}")
            for idx in valid_indices:
                if results[idx] is None:
                    results[idx] = self._fallback_predict(texts[idx])

        return [r if r is not None else self._empty_result() for r in results]

    # ── Fallback Lexicon Plutchik ────────────────────────────────────────

    def _fallback_predict(self, text: str) -> dict:
        """
        Prediksi emosi berbasis lexicon Plutchik sebagai fallback.

        Menggunakan pencocokan kata dari kamus emosi Bahasa Indonesia.
        Jika tidak ada emosi terdeteksi, kembalikan 'Netral'.

        Args:
            text: Teks yang akan dianalisis

        Returns:
            Dictionary berisi label emosi, confidence, dan skor per emosi
        """
        if not text or not isinstance(text, str) or not text.strip():
            return self._empty_result()

        cleaned = preprocess_text(text)
        if not cleaned.strip():
            cleaned = text

        words = cleaned.lower().split()
        if not words:
            return self._empty_result()

        word_count = max(len(words), 1)
        raw_scores: dict[str, float] = defaultdict(float)

        for word in words:
            if word in self._word_index:
                for emotion in self._word_index[word]:
                    raw_scores[emotion] += _EMOTION_WEIGHTS.get(emotion, 1.0)

        # Normalisasi skor per emosi
        scores: dict[str, float] = {}
        for emotion in self.labels:
            if emotion == "Netral":
                continue
            raw = raw_scores.get(emotion, 0.0) / word_count
            scores[emotion] = min(1.0, round(raw * 5, 4))

        # Tentukan emosi dominan
        non_zero_scores = {e: s for e, s in scores.items() if s > 0}

        if not non_zero_scores:
            # Tidak ada emosi terdeteksi → Netral
            result_scores = {e: 0.0 for e in self.labels}
            result_scores["Netral"] = 1.0
            return {
                "emotion": "Netral",
                "confidence": 0.5,
                "scores": result_scores,
            }

        dominant_emotion = max(non_zero_scores, key=non_zero_scores.get)  # type: ignore[arg-type]
        dominant_score = non_zero_scores[dominant_emotion]

        # Hitung confidence berdasarkan dominansi relatif
        total_score = sum(non_zero_scores.values())
        confidence = dominant_score / total_score if total_score > 0 else 0.0
        confidence = min(1.0, confidence)

        # Distribusi probabilitas dengan softmax-like normalisasi
        all_scores_sum = sum(scores.values())
        result_scores: dict[str, float] = {}
        for emotion in self.labels:
            if emotion == "Netral":
                if all_scores_sum == 0:
                    result_scores["Netral"] = 1.0
                else:
                    # Skor Netral menurun saat emosi lain kuat
                    neutral_score = max(0.0, 1.0 - min(1.0, all_scores_sum))
                    result_scores["Netral"] = round(neutral_score, 4)
            else:
                result_scores[emotion] = scores.get(emotion, 0.0)

        return {
            "emotion": dominant_emotion,
            "confidence": round(confidence, 4),
            "scores": result_scores,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _empty_result(self) -> dict:
        """Hasil default untuk teks kosong atau tidak valid."""
        scores = {e: 0.0 for e in self.labels}
        scores["Netral"] = 1.0
        return {
            "emotion": "Netral",
            "confidence": 0.0,
            "scores": scores,
        }

    def is_model_loaded(self) -> bool:
        """Cek apakah model Transformer berhasil dimuat."""
        return self.model is not None and not self._use_fallback

    def __repr__(self) -> str:
        mode = "Transformer" if self.is_model_loaded() else "Lexicon Fallback"
        return f"<EmotionService mode={mode} device={self.device}>"
