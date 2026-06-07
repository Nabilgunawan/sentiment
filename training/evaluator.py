"""
training/evaluator.py
---------------------
Modul evaluasi model klasifikasi teks untuk platform Analisis Sentimen Media
Sosial Indonesia.

Kelas ``ModelEvaluator`` menyediakan:
- Evaluasi metrik lengkap (accuracy, precision, recall, F1)
- Confusion matrix dan classification report
- Data ROC curve (FPR, TPR, AUC per kelas)
- Rendering chart (confusion matrix, ROC curve, training history) sebagai
  PNG bytes — siap ditampilkan di frontend atau disimpan ke laporan.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # backend non-interaktif untuk server
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from training.data_utils import create_dataset, get_label_mapping

logger = logging.getLogger(__name__)

# ─── Konfigurasi model per tipe ──────────────────────────────────────────────

_MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "sentiment": {
        "labels": config.SENTIMENT_LABELS,
        "save_dir": config.MODELS_SENTIMENT_DIR,
    },
    "emotion": {
        "labels": config.EMOTION_LABELS,
        "save_dir": config.MODELS_EMOTION_DIR,
    },
    "sarcasm": {
        "labels": config.SARCASM_LABELS,
        "save_dir": config.MODELS_SARCASM_DIR,
    },
}

# ─── Palet warna untuk chart ─────────────────────────────────────────────────

_CHART_COLORS: dict[str, list[str]] = {
    "sentiment": ["#4CAF50", "#9E9E9E", "#F44336"],          # hijau, abu, merah
    "emotion": [
        "#F44336",  # Marah – merah
        "#FFD600",  # Senang – kuning
        "#2196F3",  # Sedih – biru
        "#9C27B0",  # Takut – ungu
        "#795548",  # Jijik – coklat
        "#FF9800",  # Terkejut – oranye
        "#9E9E9E",  # Netral – abu
    ],
    "sarcasm": ["#4CAF50", "#FF5722"],                       # hijau, oranye-merah
}

_DEFAULT_PALETTE = [
    "#4CAF50", "#2196F3", "#F44336", "#FF9800", "#9C27B0",
    "#795548", "#9E9E9E", "#00BCD4", "#E91E63", "#CDDC39",
]


class ModelEvaluator:
    """Evaluator untuk model klasifikasi teks yang sudah dilatih.

    Args:
        model_type: Jenis model – ``'sentiment'``, ``'emotion'``, atau
            ``'sarcasm'``.

    Raises:
        ValueError: Jika ``model_type`` tidak valid.
        FileNotFoundError: Jika model belum ditemukan di direktori penyimpanan.
    """

    def __init__(self, model_type: str) -> None:
        model_type = model_type.lower().strip()
        if model_type not in _MODEL_CONFIG:
            raise ValueError(
                f"model_type '{model_type}' tidak valid. "
                f"Pilih salah satu: {list(_MODEL_CONFIG.keys())}"
            )

        self.model_type = model_type
        cfg = _MODEL_CONFIG[model_type]
        self.labels: list[str] = cfg["labels"]
        self.save_dir: Path = Path(cfg["save_dir"])
        self.device: str = config.resolve_device()

        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

        self._load_model()

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Memuat model dan tokenizer dari direktori penyimpanan."""
        model_path = self.save_dir

        if not model_path.exists() or not any(model_path.iterdir()):
            raise FileNotFoundError(
                f"Model '{self.model_type}' tidak ditemukan di '{model_path}'. "
                "Latih model terlebih dahulu."
            )

        logger.info("Memuat model '%s' dari '%s' …", self.model_type, model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        )
        self.model.to(self.device)
        self.model.eval()

        # Muat label mapping
        mapping_path = model_path / "label_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            self.label2id = mapping.get("label2id", {})
            self.id2label = {int(k): v for k, v in mapping.get("id2label", {}).items()}
            logger.info("Label mapping dimuat: %s", list(self.label2id.keys()))
        else:
            # Fallback: gunakan config model
            self.label2id = {lbl: i for i, lbl in enumerate(self.labels)}
            self.id2label = {i: lbl for lbl, i in self.label2id.items()}
            logger.warning(
                "label_mapping.json tidak ditemukan, menggunakan label default."
            )

    # ── Prediction ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _predict(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> tuple[list[str], np.ndarray]:
        """Melakukan prediksi pada daftar teks.

        Returns:
            Tuple (predicted_labels, probabilities_matrix).
        """
        assert self.model is not None
        assert self.tokenizer is not None

        dataset = create_dataset(
            texts=texts,
            labels=["_"] * len(texts),  # dummy labels
            tokenizer=self.tokenizer,
            max_length=config.MAX_SEQ_LENGTH,
            label2id={"_": 0},  # dummy mapping
        )

        # Kita perlu membypass label encoding di __getitem__,
        # jadi kita tokenisasi secara manual
        all_preds: list[int] = []
        all_probs: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encodings = self.tokenizer(
                batch_texts,
                max_length=config.MAX_SEQ_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds.tolist())
            all_probs.append(probs)

        probs_matrix = np.vstack(all_probs)
        predicted_labels = [self.id2label.get(p, f"UNKNOWN_{p}") for p in all_preds]

        return predicted_labels, probs_matrix

    # ── Core Metrics ──────────────────────────────────────────────────────────

    @staticmethod
    def get_accuracy(y_true: list[str], y_pred: list[str]) -> float:
        """Menghitung akurasi klasifikasi.

        Args:
            y_true: Label sebenarnya.
            y_pred: Label prediksi.

        Returns:
            Nilai akurasi (0.0 – 1.0).
        """
        return float(accuracy_score(y_true, y_pred))

    @staticmethod
    def get_precision_recall_f1(
        y_true: list[str],
        y_pred: list[str],
    ) -> dict[str, Any]:
        """Menghitung precision, recall, dan F1-score per kelas serta
        rata-rata macro dan weighted.

        Returns:
            Dictionary dengan kunci ``precision``, ``recall``, ``f1``,
            masing-masing berisi ``macro``, ``weighted``, dan ``per_class``.
        """
        labels_sorted = sorted(set(y_true) | set(y_pred))

        result: dict[str, Any] = {}
        for metric_name, metric_fn in [
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]:
            macro = float(
                metric_fn(y_true, y_pred, labels=labels_sorted, average="macro", zero_division=0)
            )
            weighted = float(
                metric_fn(y_true, y_pred, labels=labels_sorted, average="weighted", zero_division=0)
            )
            per_class_values = metric_fn(
                y_true, y_pred, labels=labels_sorted, average=None, zero_division=0
            )
            per_class = {
                lbl: round(float(val), 6)
                for lbl, val in zip(labels_sorted, per_class_values)
            }
            result[metric_name] = {
                "macro": round(macro, 6),
                "weighted": round(weighted, 6),
                "per_class": per_class,
            }
        return result

    @staticmethod
    def get_confusion_matrix(
        y_true: list[str],
        y_pred: list[str],
    ) -> dict[str, Any]:
        """Menghitung confusion matrix.

        Returns:
            Dictionary dengan ``matrix`` (list of lists) dan ``labels``.
        """
        labels_sorted = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        return {
            "matrix": cm.tolist(),
            "labels": labels_sorted,
        }

    @staticmethod
    def get_classification_report(
        y_true: list[str],
        y_pred: list[str],
    ) -> str:
        """Menghasilkan classification report dari sklearn.

        Returns:
            String laporan klasifikasi.
        """
        return classification_report(y_true, y_pred, zero_division=0)

    @staticmethod
    def get_roc_data(
        y_true: list[str],
        y_probs: np.ndarray,
        labels: list[str],
    ) -> dict[str, Any]:
        """Menghitung data ROC curve (FPR, TPR, AUC) per kelas.

        Mendukung multiclass melalui pendekatan one-vs-rest.

        Args:
            y_true: Label sebenarnya.
            y_probs: Matriks probabilitas (n_samples × n_classes).
            labels: Daftar label sesuai urutan kolom probabilitas.

        Returns:
            Dictionary per kelas berisi ``fpr``, ``tpr``, ``auc``.
        """
        if len(labels) < 2:
            logger.warning("ROC curve membutuhkan minimal 2 kelas.")
            return {}

        y_true_bin = label_binarize(y_true, classes=labels)

        # Untuk kasus biner, label_binarize mengembalikan 1 kolom
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

        roc_data: dict[str, Any] = {}
        for i, label in enumerate(labels):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                auc_val = float(roc_auc_score(y_true_bin[:, i], y_probs[:, i]))
                roc_data[label] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "auc": round(auc_val, 6),
                }
            except Exception as exc:
                logger.warning(
                    "Gagal menghitung ROC untuk kelas '%s': %s", label, exc
                )
                roc_data[label] = {"fpr": [], "tpr": [], "auc": 0.0}

        return roc_data

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        texts: list[str],
        labels: list[str],
    ) -> dict[str, Any]:
        """Evaluasi lengkap model pada data yang diberikan.

        Args:
            texts: Daftar teks untuk dievaluasi.
            labels: Label sebenarnya.

        Returns:
            Dictionary berisi semua metrik evaluasi.
        """
        if not texts or not labels:
            raise ValueError("texts dan labels tidak boleh kosong.")

        logger.info(
            "Mengevaluasi model '%s' pada %d sampel …",
            self.model_type,
            len(texts),
        )

        y_pred, y_probs = self._predict(texts)
        y_true = labels

        accuracy = self.get_accuracy(y_true, y_pred)
        prf = self.get_precision_recall_f1(y_true, y_pred)
        cm = self.get_confusion_matrix(y_true, y_pred)
        report = self.get_classification_report(y_true, y_pred)

        # ROC data — gunakan label dari model
        model_labels = [self.id2label[i] for i in sorted(self.id2label.keys())]
        roc_data = self.get_roc_data(y_true, y_probs, model_labels)

        result: dict[str, Any] = {
            "accuracy": round(accuracy, 6),
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "confusion_matrix": cm,
            "classification_report": report,
            "roc_data": roc_data,
        }

        logger.info(
            "Evaluasi selesai: accuracy=%.4f, macro_f1=%.4f",
            accuracy,
            prf["f1"]["macro"],
        )
        return result

    # ── Full Report ───────────────────────────────────────────────────────────

    def full_report(
        self,
        texts: list[str],
        labels: list[str],
    ) -> dict[str, Any]:
        """Laporan evaluasi lengkap termasuk chart dalam bentuk PNG bytes.

        Args:
            texts: Daftar teks untuk dievaluasi.
            labels: Label sebenarnya.

        Returns:
            Dictionary metrik + ``charts`` berisi PNG bytes.
        """
        metrics = self.evaluate(texts, labels)

        charts: dict[str, bytes] = {}

        try:
            charts["confusion_matrix"] = self.render_confusion_matrix_chart(
                metrics["confusion_matrix"]
            )
        except Exception as exc:
            logger.warning("Gagal merender confusion matrix chart: %s", exc)

        try:
            if metrics.get("roc_data"):
                charts["roc_curve"] = self.render_roc_curve_chart(
                    metrics["roc_data"]
                )
        except Exception as exc:
            logger.warning("Gagal merender ROC curve chart: %s", exc)

        metrics["charts"] = charts
        return metrics

    # ── Chart Rendering ───────────────────────────────────────────────────────

    def _get_colors(self, n: int | None = None) -> list[str]:
        """Mengambil palet warna untuk model_type ini."""
        colors = _CHART_COLORS.get(self.model_type, _DEFAULT_PALETTE)
        if n is not None and n > len(colors):
            # Extend palet jika kurang
            extra = _DEFAULT_PALETTE * ((n // len(_DEFAULT_PALETTE)) + 1)
            colors = (colors + extra)[:n]
        return colors

    def render_confusion_matrix_chart(
        self,
        cm_data: dict[str, Any],
    ) -> bytes:
        """Merender confusion matrix sebagai heatmap PNG.

        Args:
            cm_data: Dictionary berisi ``matrix`` dan ``labels``.

        Returns:
            PNG bytes dari chart.
        """
        matrix = np.array(cm_data["matrix"])
        labels_list = cm_data["labels"]

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 6))

        cmap = plt.cm.Blues  # type: ignore[attr-defined]
        im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(labels_list)))
        ax.set_yticks(range(len(labels_list)))
        ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels_list, fontsize=10)

        # Anotasi angka di setiap sel
        thresh = matrix.max() / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                color = "white" if matrix[i, j] > thresh else "black"
                ax.text(
                    j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold",
                )

        ax.set_xlabel("Label Prediksi", fontsize=12)
        ax.set_ylabel("Label Sebenarnya", fontsize=12)
        ax.set_title(
            f"Confusion Matrix — {self.model_type.capitalize()}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        return self._fig_to_bytes(fig)

    def render_roc_curve_chart(
        self,
        roc_data: dict[str, Any],
    ) -> bytes:
        """Merender ROC curve per kelas sebagai PNG.

        Args:
            roc_data: Dictionary per kelas berisi ``fpr``, ``tpr``, ``auc``.

        Returns:
            PNG bytes dari chart.
        """
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(8, 6))

        colors = self._get_colors(len(roc_data))

        for idx, (label, data) in enumerate(roc_data.items()):
            fpr = data.get("fpr", [])
            tpr = data.get("tpr", [])
            auc_val = data.get("auc", 0.0)

            if not fpr or not tpr:
                continue

            color = colors[idx % len(colors)]
            ax.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f"{label} (AUC = {auc_val:.3f})",
            )

        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", alpha=0.7)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            f"ROC Curve — {self.model_type.capitalize()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()

        return self._fig_to_bytes(fig)

    @staticmethod
    def render_training_history_chart(
        history: list[dict[str, Any]],
    ) -> bytes:
        """Merender chart loss dan akurasi pelatihan sebagai PNG.

        Args:
            history: Daftar dictionary per epoch berisi ``epoch``,
                ``train_loss``, ``val_loss``, ``val_accuracy``.

        Returns:
            PNG bytes dari chart.
        """
        if not history:
            raise ValueError("history tidak boleh kosong.")

        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        val_accs = [h["val_accuracy"] for h in history]

        plt.style.use("dark_background")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(epochs, train_losses, "o-", color="#2196F3", lw=2, label="Train Loss")
        ax1.plot(epochs, val_losses, "o-", color="#F44336", lw=2, label="Val Loss")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Loss per Epoch", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Accuracy plot
        ax2.plot(epochs, val_accs, "o-", color="#4CAF50", lw=2, label="Val Accuracy")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_title("Validation Accuracy per Epoch", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.0, 1.05])

        fig.tight_layout()

        return ModelEvaluator._fig_to_bytes(fig)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_bytes(fig: plt.Figure) -> bytes:
        """Mengonversi matplotlib Figure menjadi PNG bytes dan menutup figure.

        Args:
            fig: Instance matplotlib Figure.

        Returns:
            Bytes PNG.
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()
