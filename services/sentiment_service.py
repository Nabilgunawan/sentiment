"""
services/sentiment_service.py
------------------------------
Layanan analisis sentimen Bahasa Indonesia menggunakan model Transformer (IndoBERT).
Mendukung model fine-tuned, pre-trained, dan fallback ke lexicon-based analysis.

Fitur utama:
- Load model fine-tuned dari path lokal, jika tidak ada fallback ke pre-trained
- Jika model gagal dimuat sepenuhnya, gunakan fallback lexicon
- Prediksi tunggal dan batch dengan dynamic batching
- Softmax probability distribution untuk setiap label
- Penanganan edge case: teks kosong, None, model tidak tersedia
"""

import re
import logging
from typing import Optional
from pathlib import Path

import config
from preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# ── Lexicon Kata Positif & Negatif Bahasa Indonesia ──────────────────────────

POSITIVE_WORDS: set[str] = {
    "baik", "bagus", "hebat", "mantap", "keren", "senang", "gembira",
    "bahagia", "suka", "puas", "bangga", "bersyukur", "semangat", "optimis",
    "mudah", "nyaman", "enak", "berhasil", "sukses", "terbaik",
    "luar biasa", "fantastis", "sempurna", "ramah", "damai", "cerdas",
    "kreatif", "inovatif", "produktif", "tegas", "adil", "recommended",
    "rekomendasi", "sip", "siap", "amazing", "great", "nice", "good",
    "best", "perfect", "lengkap", "jelas", "transparan", "lancar", "praktis",
}

NEGATIVE_WORDS: set[str] = {
    "buruk", "jelek", "parah", "sedih", "kecewa", "kesal", "marah",
    "benci", "gagal", "rugi", "bodoh", "susah", "sulit", "payah",
    "lambat", "lemot", "lemah", "kurang", "mahal", "boros", "sakit",
    "sial", "celaka", "krisis", "korupsi", "kejahatan", "konflik",
    "sampah", "kotor", "bohong", "curang", "hancur", "rusak", "menyesal",
    "stress", "depresi", "cemas", "takut", "error", "bug", "crash",
}

# ── Kata negasi dan intensifier ──────────────────────────────────────────────

_NEGATION_WORDS: set[str] = {
    "tidak", "nggak", "gak", "ga", "bukan", "belum", "jangan",
    "tak", "tanpa", "tiada", "engga", "enggak", "kagak", "ndak",
}

_INTENSIFIER_WORDS: set[str] = {
    "sangat", "sekali", "banget", "amat", "benar-benar", "sungguh",
    "luar biasa", "super", "paling", "terlalu", "betul-betul",
}


class SentimentService:
    """
    Layanan analisis sentimen berbasis Transformer dengan fallback lexicon.

    Alur inisialisasi:
    1. Coba load model fine-tuned dari SENTIMENT_FINETUNED_PATH
    2. Jika tidak ada, load pre-trained dari SENTIMENT_MODEL_NAME
    3. Jika semua gagal, gunakan fallback lexicon-based

    Attributes:
        model: Model Transformer untuk klasifikasi sentimen
        tokenizer: Tokenizer yang cocok dengan model
        device: Device untuk inference (cuda/cpu)
        labels: Daftar label sentimen [Positive, Neutral, Negative]
    """

    def __init__(self) -> None:
        """Inisialisasi SentimentService dengan strategi loading bertingkat."""
        self.model = None
        self.tokenizer = None
        self.device: str = config.resolve_device()
        self.labels: list[str] = config.SENTIMENT_LABELS
        self._use_fallback: bool = False

        self._load_model()

    # ── Model Loading ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Muat model Transformer dengan strategi bertingkat:
        fine-tuned → pre-trained → fallback lexicon.
        """
        if not config.USE_TRANSFORMERS:
            logger.info("USE_TRANSFORMERS set to False. Menggunakan fallback lexicon-based secara instan.")
            self._use_fallback = True
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            logger.warning(
                "Library transformers/torch tidak terinstall. "
                "Menggunakan fallback lexicon-based."
            )
            self._use_fallback = True
            return

        # Strategi 1: Load model fine-tuned
        finetuned_path = Path(config.SENTIMENT_FINETUNED_PATH)
        if finetuned_path.exists() and self._is_valid_model_dir(finetuned_path):
            try:
                logger.info(
                    f"Memuat model sentimen fine-tuned dari: {finetuned_path}"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(str(finetuned_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(finetuned_path)
                )
                self._move_model_to_device()
                logger.info("Model sentimen fine-tuned berhasil dimuat.")
                return
            except Exception as e:
                logger.warning(
                    f"Gagal memuat model fine-tuned: {e}. "
                    "Mencoba model pre-trained..."
                )

        # Cek jika model pre-trained adalah raw language model tanpa classification head terlatih.
        # Menggunakan fallback lexicon-based jika model-nya raw, untuk mencegah prediksi acak.
        raw_models = ["indobenchmark/indobert-base-p1", "indobenchmark/indobert-base-p2", "indobenchmark/indobert-large-p1", "bert-base-multilingual-cased"]
        if config.SENTIMENT_MODEL_NAME in raw_models:
            logger.info(
                f"Model pre-trained '{config.SENTIMENT_MODEL_NAME}' adalah raw model tanpa head klasifikasi terlatih. "
                "Menggunakan fallback lexicon-based untuk akurasi dasar (sebelum model di-training)."
            )
            self._use_fallback = True
            logger.info("SentimentService diinisialisasi dengan fallback lexicon.")
            return

        # Strategi 2: Load model pre-trained dari HuggingFace
        try:
            model_name = config.SENTIMENT_MODEL_NAME
            logger.info(f"Memuat model sentimen pre-trained: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(self.labels)
            )
            self._move_model_to_device()
            logger.info("Model sentimen pre-trained berhasil dimuat.")
            return
        except Exception as e:
            logger.warning(
                f"Gagal memuat model pre-trained: {e}. "
                "Menggunakan fallback lexicon-based."
            )

        # Strategi 3: Fallback ke lexicon
        self._use_fallback = True
        logger.info("SentimentService diinisialisasi dengan fallback lexicon.")

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
            logger.info(f"Model sentimen dipindahkan ke device: {self.device}")

    # ── Prediksi Tunggal ─────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Prediksi sentimen untuk satu teks.

        Args:
            text: Teks mentah yang akan dianalisis

        Returns:
            Dictionary berisi:
                - sentiment (str): Label sentimen (Positive/Neutral/Negative)
                - confidence (float): Skor kepercayaan prediksi (0.0-1.0)
                - prob_positive (float): Probabilitas positif
                - prob_neutral (float): Probabilitas netral
                - prob_negative (float): Probabilitas negatif
        """
        # Penanganan teks kosong / None
        if not text or not isinstance(text, str) or not text.strip():
            return self._empty_result()

        if self._use_fallback:
            return self._fallback_predict(text)

        try:
            return self._model_predict(text)
        except Exception as e:
            logger.error(f"Error pada prediksi model sentimen: {e}")
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

        return {
            "sentiment": label,
            "confidence": round(confidence, 4),
            "prob_positive": round(float(probs_np[0]), 4),
            "prob_neutral": round(float(probs_np[1]), 4),
            "prob_negative": round(float(probs_np[2]), 4),
        }

    # ── Prediksi Batch ───────────────────────────────────────────────────

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Prediksi sentimen untuk batch teks dengan dynamic batching.

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

        # Preprocess semua teks dalam batch
        cleaned_texts = [preprocess_text(t) if t and isinstance(t, str) else "" for t in texts]

        # Pisahkan teks kosong dan non-kosong
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

                results[orig_idx] = {
                    "sentiment": label,
                    "confidence": round(confidence, 4),
                    "prob_positive": round(float(p[0]), 4),
                    "prob_neutral": round(float(p[1]), 4),
                    "prob_negative": round(float(p[2]), 4),
                }
        except Exception as e:
            logger.error(f"Error pada batch prediksi sentimen: {e}")
            for idx in valid_indices:
                if results[idx] is None:
                    results[idx] = self._fallback_predict(texts[idx])

        return [r if r is not None else self._empty_result() for r in results]

    # ── Fallback Lexicon-Based ───────────────────────────────────────────

    def _fallback_predict(self, text: str) -> dict:
        """
        Prediksi sentimen berbasis lexicon sebagai fallback.

        Menggunakan daftar kata positif/negatif Bahasa Indonesia,
        dengan dukungan negasi dan intensifier.

        Args:
            text: Teks yang akan dianalisis

        Returns:
            Dictionary hasil prediksi sentimen
        """
        if not text or not isinstance(text, str) or not text.strip():
            return self._empty_result()

        cleaned = preprocess_text(text)
        if not cleaned.strip():
            cleaned = text

        words = cleaned.lower().split()
        if not words:
            return self._empty_result()

        score: float = 0.0
        word_count: int = 0
        negation_active: bool = False
        negation_span: int = 0

        for i, word in enumerate(words):
            # Cek apakah kata negasi
            if word in _NEGATION_WORDS:
                negation_active = True
                negation_span = 2
                continue

            # Hitung skor sentimen
            word_score: float = 0.0
            if word in POSITIVE_WORDS:
                word_score = 1.0
            elif word in NEGATIVE_WORDS:
                word_score = -1.0
            else:
                # Kurangi span negasi bahkan jika kata bukan lexicon
                if negation_active:
                    negation_span -= 1
                    if negation_span <= 0:
                        negation_active = False
                continue

            # Terapkan negasi
            if negation_active and negation_span > 0:
                word_score *= -0.5
                negation_span -= 1
                if negation_span <= 0:
                    negation_active = False

            # Terapkan intensifier dari kata sebelumnya
            if i > 0 and words[i - 1] in _INTENSIFIER_WORDS:
                word_score *= 1.5

            score += word_score
            word_count += 1

        if word_count == 0:
            return self._empty_result()

        # Normalisasi skor
        avg_score = score / word_count
        normalized = max(-1.0, min(1.0, avg_score))

        # Tentukan label
        if normalized > 0.15:
            label = "Positive"
        elif normalized < -0.15:
            label = "Negative"
        else:
            label = "Neutral"

        # Hitung confidence berdasarkan magnitude dan jumlah kata
        magnitude = abs(normalized)
        length_factor = min(1.0, word_count / 8)
        confidence = min(1.0, magnitude * 1.5 + length_factor * 0.2)
        if label == "Neutral":
            confidence = max(0.3, min(0.7, confidence))

        # Distribusi probabilitas (softmax-like)
        pos_raw = max(0.0, normalized)
        neg_raw = max(0.0, -normalized)
        neu_raw = 1.0 - pos_raw - neg_raw
        total = pos_raw + neg_raw + neu_raw
        if total <= 0:
            total = 1.0

        prob_pos = pos_raw / total
        prob_neg = neg_raw / total
        prob_neu = neu_raw / total

        return {
            "sentiment": label,
            "confidence": round(confidence, 4),
            "prob_positive": round(prob_pos, 4),
            "prob_neutral": round(prob_neu, 4),
            "prob_negative": round(prob_neg, 4),
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> dict:
        """Hasil default untuk teks kosong atau tidak valid."""
        return {
            "sentiment": "Neutral",
            "confidence": 0.0,
            "prob_positive": 0.0,
            "prob_neutral": 1.0,
            "prob_negative": 0.0,
        }

    def is_model_loaded(self) -> bool:
        """Cek apakah model Transformer berhasil dimuat."""
        return self.model is not None and not self._use_fallback

    def __repr__(self) -> str:
        mode = "Transformer" if self.is_model_loaded() else "Lexicon Fallback"
        return f"<SentimentService mode={mode} device={self.device}>"
