"""
services/sarcasm_service.py
----------------------------
Layanan deteksi sarkasme Bahasa Indonesia menggunakan model Transformer (IndoBERT).
Mendukung fallback ke rule-based detection dengan pola regex.

Fitur utama:
- Load model fine-tuned dari path lokal, fallback ke rule-based
- Deteksi pola sarkasme khas Indonesia (pujian + keluhan, ironi, kontradiksi)
- Pola khusus sarkasme politik dan sosial media Indonesia
- Prediksi tunggal dan batch dengan dynamic batching
"""

import re
import logging
from typing import Optional
from pathlib import Path

import config
from preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# ── Pola Regex Sarkasme Bahasa Indonesia ─────────────────────────────────────
# Setiap tuple: (pattern_regex, deskripsi_singkat)
# Pattern disusun dari paling spesifik ke paling umum.

SARCASM_PATTERNS: list[tuple[str, str]] = [
    # ── Pujian + kata negatif eksplisit ──────────────────────────────────
    (
        r"\bbagus\s+banget\b.*\b(?:error|gagal|rusak|tidak|nggak|ga\s+bisa|lemot|down|mati|mati\s+total|parah|jelek|hancur)\b",
        "Pujian 'bagus banget' diikuti kata negatif",
    ),
    (
        r"\bkeren\b.*\b(?:gagal|error|nggak|tidak|rusak|macet|down|mati|parah|jelek|hancur)\b",
        "Pujian 'keren' diikuti kata negatif",
    ),
    (
        r"\bmantap\b.*\b(?:error|gagal|tidak|nggak|rusak|lemot|down|mati|parah|jelek|hancur)\b",
        "Pujian 'mantap' diikuti kata negatif",
    ),
    (
        r"\bhebat\b.*\b(?:rusak|gagal|error|tidak|nggak|parah|down|mati|jelek|hancur)\b",
        "Pujian 'hebat' diikuti kata negatif",
    ),
    (
        r"\bsukses\b.*\b(?:gagal|error|nggak|tidak|hancur|rusak|down|mati|parah|jelek)\b",
        "Pujian 'sukses' diikuti kata negatif",
    ),
    (
        r"\bluar\s+biasa\b.*\b(?:gagal|rusak|error|parah|jelek|buruk|down|mati|hancur)\b",
        "Pujian 'luar biasa' diikuti kata negatif",
    ),
    (
        r"\bsempurna\b.*\b(?:gagal|rusak|error|hancur|jelek|down|mati|parah|buruk)\b",
        "Pujian 'sempurna' diikuti kata negatif",
    ),
    # ── Pujian berlebihan + marker keluhan ───────────────────────────────
    (
        r"\bkeren\s+banget\b.*\b(?:sampai|kok|tapi|tetapi)\b",
        "Pujian berlebihan dengan marker kontradiksi",
    ),
    (
        r"\bsangat\s+(?:baik|bagus|hebat)\b.*\b(?:tetapi|tapi|namun|sayangnya)\b",
        "Sangat positif lalu kontradiksi",
    ),
    (
        r"\bmemang\s+(?:bagus|hebat|keren)\b.*\b(?:tapi|tetapi|namun)\b",
        "Memang + pujian + kontradiksi",
    ),
    (
        r"\b(?:mantap|keren|bagus).*?(?:tapi|tetapi|sayangnya|namun)\b",
        "Pujian lalu kontradiksi",
    ),
    # ── Kepastian sarkastik ──────────────────────────────────────────────
    (
        r"\b(?:tentu|pasti|tentulah)\s+(?:sangat|sekali)\b.*\b(?:error|gagal|rusak|parah)\b",
        "Kepastian sarkastik diikuti kegagalan",
    ),
    # ── Ucapan terima kasih ironis ───────────────────────────────────────
    (
        r"\bterima\s+kasih\b.*\b(?:error|rusak|gagal|lemot|macet|parah|jelek|lambat)\b",
        "Terima kasih ironis",
    ),
    (
        r"\b(?:makasih|thx|thanks)\b.*\b(?:error|rusak|gagal|lemot|macet|parah)\b",
        "Makasih/thanks ironis",
    ),
    # ── Gampang + kegagalan ──────────────────────────────────────────────
    (
        r"\bgampang\b.*\b(?:error|gagal|nggak|tidak|susah|ribet)\b",
        "Gampang sarkastik",
    ),
    # ── Wow/waah + negatif ───────────────────────────────────────────────
    (
        r"\b(?:wow|waah|wah)\b.*\b(?:parah|jelek|buruk|payah|gagal|rusak)\b",
        "Eksklamasi positif + negatif",
    ),
    # ── Sarkasme Politik & Sosial ────────────────────────────────────────
    (
        r"\bhebat\s+(?:sekali\s+)?(?:pemerintah|pemimpin|pejabat|DPR|menteri)\b.*"
        r"\b(?:harga\s+naik|mahal|susah|korupsi|gagal|macet)\b",
        "Sarkasme terhadap pemerintah/pejabat",
    ),
    (
        r"\b(?:cerdas|pintar|brilian)\s+(?:sekali\s+)?(?:kebijakan|keputusan|aturan)\b.*"
        r"\b(?:rakyat|susah|sengsara|rugi|gagal)\b",
        "Sarkasme kebijakan",
    ),
    (
        r"\b(?:rakyat|masyarakat)\s+(?:senang|bahagia|sejahtera)\b.*"
        r"\b(?:harga\s+naik|mahal|susah|miskin|lapar)\b",
        "Sarkasme kesejahteraan rakyat",
    ),
    (
        r"\b(?:adil|transparan|jujur)\s+(?:sekali|banget)\b.*"
        r"\b(?:korupsi|curang|bohong|nepotisme)\b",
        "Sarkasme keadilan/transparansi",
    ),
    # ── Sarkasme layanan/produk ──────────────────────────────────────────
    (
        r"\b(?:pelayanan|layanan|service)\s+(?:terbaik|bagus|mantap)\b.*"
        r"\b(?:lama|lambat|ribet|susah|antri|antre)\b",
        "Sarkasme layanan",
    ),
    (
        r"\b(?:canggih|modern|maju)\s+(?:sekali|banget)\b.*"
        r"\b(?:error|crash|lemot|hang|down|rusak)\b",
        "Sarkasme teknologi",
    ),
    # ── Pola tanda baca sarkastik ────────────────────────────────────────
    (
        r"\b(?:bagus|keren|mantap|hebat)\b\s*\.{3,}",
        "Pujian diikuti ellipsis (indikator ironi)",
    ),
    (
        r"(?:ya\s+)?(?:bagus|keren|mantap|hebat)\s+(?:deh|lah|kok)\b",
        "Pujian dengan partikel sarkastik",
    ),
]


class SarcasmService:
    """
    Layanan deteksi sarkasme berbasis Transformer dengan fallback rule-based.

    Alur inisialisasi:
    1. Coba load model fine-tuned dari SARCASM_FINETUNED_PATH
    2. Jika tidak ada model fine-tuned, coba load pre-trained
    3. Jika semua gagal, gunakan fallback rule-based (regex)

    Attributes:
        model: Model Transformer untuk klasifikasi sarkasme
        tokenizer: Tokenizer yang cocok dengan model
        device: Device untuk inference (cuda/cpu)
        labels: Daftar label ['Not Sarcasm', 'Sarcasm']
    """

    def __init__(self) -> None:
        """Inisialisasi SarcasmService dengan strategi loading bertingkat."""
        self.model = None
        self.tokenizer = None
        self.device: str = config.resolve_device()
        self.labels: list[str] = config.SARCASM_LABELS
        self._use_fallback: bool = False

        # Compile semua regex pattern satu kali saat init
        self._compiled_patterns: list[tuple[re.Pattern[str], str]] = []
        for pattern_str, desc in SARCASM_PATTERNS:
            try:
                self._compiled_patterns.append(
                    (re.compile(pattern_str, re.IGNORECASE), desc)
                )
            except re.error as e:
                logger.warning(f"Gagal compile pattern sarkasme '{desc}': {e}")

        self._load_model()

    # ── Model Loading ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """
        Muat model Transformer dengan strategi bertingkat:
        fine-tuned → pre-trained → fallback rule-based.
        """
        if not config.USE_TRANSFORMERS:
            logger.info("USE_TRANSFORMERS set to False. Menggunakan fallback rule-based secara instan untuk sarkasme.")
            self._use_fallback = True
            return

        try:
            import torch  # noqa: F401
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            logger.warning(
                "Library transformers/torch tidak terinstall. "
                "Menggunakan fallback rule-based untuk sarkasme."
            )
            self._use_fallback = True
            return

        # Strategi 1: Load model fine-tuned
        finetuned_path = Path(config.SARCASM_FINETUNED_PATH)
        if finetuned_path.exists() and self._is_valid_model_dir(finetuned_path):
            try:
                logger.info(
                    f"Memuat model sarkasme fine-tuned dari: {finetuned_path}"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(str(finetuned_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(finetuned_path)
                )
                self._move_model_to_device()
                logger.info("Model sarkasme fine-tuned berhasil dimuat.")
                return
            except Exception as e:
                logger.warning(
                    f"Gagal memuat model sarkasme fine-tuned: {e}. "
                    "Mencoba model pre-trained..."
                )

        # Cek jika model pre-trained adalah raw language model tanpa classification head terlatih.
        # Menggunakan fallback rule-based (regex) jika model-nya raw, untuk mencegah prediksi acak.
        raw_models = ["indobenchmark/indobert-base-p1", "indobenchmark/indobert-base-p2", "indobenchmark/indobert-large-p1", "bert-base-multilingual-cased"]
        if config.SARCASM_MODEL_NAME in raw_models:
            logger.info(
                f"Model pre-trained '{config.SARCASM_MODEL_NAME}' adalah raw model tanpa head klasifikasi terlatih. "
                "Menggunakan fallback rule-based untuk akurasi dasar (sebelum model di-training)."
            )
            self._use_fallback = True
            logger.info("SarcasmService diinisialisasi dengan fallback rule-based.")
            return

        # Strategi 2: Load model pre-trained
        try:
            model_name = config.SARCASM_MODEL_NAME
            logger.info(f"Memuat model sarkasme pre-trained: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=len(self.labels)
            )
            self._move_model_to_device()
            logger.info("Model sarkasme pre-trained berhasil dimuat.")
            return
        except Exception as e:
            logger.warning(
                f"Gagal memuat model sarkasme pre-trained: {e}. "
                "Menggunakan fallback rule-based."
            )

        # Strategi 3: Fallback ke rule-based
        self._use_fallback = True
        logger.info("SarcasmService diinisialisasi dengan fallback rule-based.")

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
            logger.info(f"Model sarkasme dipindahkan ke device: {self.device}")

    # ── Prediksi Tunggal ─────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Prediksi sarkasme untuk satu teks.

        Args:
            text: Teks mentah yang akan dianalisis

        Returns:
            Dictionary berisi:
                - sarcasm (bool): True jika terdeteksi sarkasme
                - confidence (float): Skor kepercayaan (0.0-1.0)
        """
        if not text or not isinstance(text, str) or not text.strip():
            return self._empty_result()

        if self._use_fallback:
            return self._fallback_predict(text)

        try:
            return self._model_predict(text)
        except Exception as e:
            logger.error(f"Error pada prediksi model sarkasme: {e}")
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

        is_sarcasm = label == "Sarcasm"

        return {
            "sarcasm": is_sarcasm,
            "confidence": round(confidence, 4),
        }

    # ── Prediksi Batch ───────────────────────────────────────────────────

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Prediksi sarkasme untuk batch teks dengan dynamic batching.

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

                results[orig_idx] = {
                    "sarcasm": label == "Sarcasm",
                    "confidence": round(confidence, 4),
                }
        except Exception as e:
            logger.error(f"Error pada batch prediksi sarkasme: {e}")
            for idx in valid_indices:
                if results[idx] is None:
                    results[idx] = self._fallback_predict(texts[idx])

        return [r if r is not None else self._empty_result() for r in results]

    # ── Fallback Rule-Based ──────────────────────────────────────────────

    def _fallback_predict(self, text: str) -> dict:
        """
        Deteksi sarkasme berbasis aturan (rule-based) sebagai fallback.

        Menggunakan pencocokan pola regex untuk mendeteksi:
        - Pujian positif diikuti konteks negatif
        - Pujian berlebihan dengan marker keluhan (tapi, tetapi, namun)
        - Ucapan terima kasih ironis
        - Sarkasme politik dan sosial khas Indonesia
        - Partikel sarkastik (deh, lah, kok)
        - Ellipsis setelah pujian

        Args:
            text: Teks yang akan dianalisis

        Returns:
            Dictionary berisi sarcasm (bool) dan confidence (float)
        """
        if not text or not isinstance(text, str) or not text.strip():
            return self._empty_result()

        # Gunakan teks asli (lowercase) untuk pattern matching karena
        # sarkasme sering bergantung pada struktur kalimat asli
        text_lower = text.lower().strip()

        matched_patterns: list[str] = []
        for pattern, desc in self._compiled_patterns:
            if pattern.search(text_lower):
                matched_patterns.append(desc)

        if not matched_patterns:
            return {
                "sarcasm": False,
                "confidence": 0.7,
            }

        # Confidence meningkat seiring jumlah pola yang cocok
        base_confidence = 0.65
        pattern_bonus = min(0.3, len(matched_patterns) * 0.1)
        confidence = min(1.0, base_confidence + pattern_bonus)

        logger.debug(
            f"Sarkasme terdeteksi ({len(matched_patterns)} pola): "
            f"{', '.join(matched_patterns[:3])}"
        )

        return {
            "sarcasm": True,
            "confidence": round(confidence, 4),
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> dict:
        """Hasil default untuk teks kosong atau tidak valid."""
        return {
            "sarcasm": False,
            "confidence": 0.0,
        }

    def is_model_loaded(self) -> bool:
        """Cek apakah model Transformer berhasil dimuat."""
        return self.model is not None and not self._use_fallback

    def __repr__(self) -> str:
        mode = "Transformer" if self.is_model_loaded() else "Rule-Based Fallback"
        return f"<SarcasmService mode={mode} device={self.device}>"
