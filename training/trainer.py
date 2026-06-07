"""
training/trainer.py
-------------------
Modul pelatihan (fine-tuning) model IndoBERT untuk tugas klasifikasi teks
sentimen, emosi, dan sarkasme dalam bahasa Indonesia.

Kelas utama ``ModelTrainer`` mendukung:
- Pemuatan base-model dari HuggingFace (IndoBERT)
- Fine-tuning dengan AdamW + linear warmup scheduler
- Early stopping berdasarkan validation loss
- Penyimpanan model terbaik (lowest val_loss)
- Callback real-time per epoch untuk integrasi UI
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

import config
from training.data_utils import (
    SentimentDataset,
    create_dataset,
    get_class_weights,
    get_label_mapping,
)

logger = logging.getLogger(__name__)

# ─── Konfigurasi model per tipe ──────────────────────────────────────────────

_MODEL_CONFIG: dict[str, dict[str, Any]] = {
    "sentiment": {
        "model_name": config.SENTIMENT_MODEL_NAME,
        "labels": config.SENTIMENT_LABELS,
        "save_dir": config.MODELS_SENTIMENT_DIR,
    },
    "emotion": {
        "model_name": config.EMOTION_MODEL_NAME,
        "labels": config.EMOTION_LABELS,
        "save_dir": config.MODELS_EMOTION_DIR,
    },
    "sarcasm": {
        "model_name": config.SARCASM_MODEL_NAME,
        "labels": config.SARCASM_LABELS,
        "save_dir": config.MODELS_SARCASM_DIR,
    },
}


class ModelTrainer:
    """Trainer untuk fine-tuning model klasifikasi teks berbasis IndoBERT.

    Args:
        model_type: Jenis model – ``'sentiment'``, ``'emotion'``, atau
            ``'sarcasm'``.

    Raises:
        ValueError: Jika ``model_type`` tidak valid.

    Example::

        trainer = ModelTrainer("sentiment")
        history = trainer.train(
            train_texts, train_labels,
            val_texts, val_labels,
            callback=lambda e, tl, vl, va: print(f"Epoch {e}: val_acc={va:.3f}"),
        )
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

        self.base_model_name: str = cfg["model_name"]
        self.labels: list[str] = cfg["labels"]
        self.save_dir: Path = Path(cfg["save_dir"])

        self.device: str = config.resolve_device()
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

        logger.info(
            "ModelTrainer diinisialisasi: type=%s, base=%s, device=%s, save=%s",
            self.model_type,
            self.base_model_name,
            self.device,
            self.save_dir,
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _setup_model(self, num_labels: int) -> None:
        """Memuat tokenizer dan model dari HuggingFace serta mengonfigurasi
        jumlah label untuk klasifikasi.

        Args:
            num_labels: Jumlah kelas/label.
        """
        logger.info(
            "Memuat tokenizer dan model '%s' dengan %d label …",
            self.base_model_name,
            num_labels,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)  # type: ignore[union-attr]
        logger.info("Model dimuat ke device '%s'.", self.device)

    # ── Training Loop ─────────────────────────────────────────────────────────

    def train(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str],
        val_labels: list[str],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        callback: Optional[Callable[[int, float, float, float], None]] = None,
    ) -> dict[str, Any]:
        """Melatih (fine-tune) model pada data yang diberikan.

        Args:
            train_texts: Teks pelatihan.
            train_labels: Label pelatihan.
            val_texts: Teks validasi.
            val_labels: Label validasi.
            epochs: Jumlah epoch (default dari config).
            batch_size: Ukuran batch (default dari config).
            learning_rate: Learning rate (default dari config).
            callback: Fungsi opsional ``callback(epoch, train_loss, val_loss,
                val_accuracy)`` yang dipanggil setelah setiap epoch.

        Returns:
            Dictionary berisi riwayat pelatihan dan metadata.
        """
        # Resolusi hyperparameter
        epochs = epochs or config.TRAINING_EPOCHS
        batch_size = batch_size or config.TRAINING_BATCH_SIZE
        learning_rate = learning_rate or config.TRAINING_LEARNING_RATE
        max_seq_length = config.TRAINING_MAX_SEQ_LENGTH

        logger.info(
            "Memulai pelatihan model '%s': epochs=%d, batch_size=%d, lr=%s",
            self.model_type,
            epochs,
            batch_size,
            learning_rate,
        )

        # Label mapping
        all_labels = train_labels + val_labels
        self.label2id, self.id2label = get_label_mapping(all_labels)
        num_labels = len(self.label2id)

        # Setup model & tokenizer
        self._setup_model(num_labels)
        assert self.model is not None
        assert self.tokenizer is not None

        # Datasets & DataLoaders
        train_dataset = create_dataset(
            train_texts, train_labels, self.tokenizer, max_seq_length, self.label2id
        )
        val_dataset = create_dataset(
            val_texts, val_labels, self.tokenizer, max_seq_length, self.label2id
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Class weights untuk loss function
        class_weights = get_class_weights(train_labels).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate
        )
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * config.TRAINING_WARMUP_RATIO)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Tracking
        history: list[dict[str, Any]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_val_accuracy = 0.0
        patience_counter = 0
        patience_limit = 2  # early stopping setelah 2 epoch val_loss naik berturut-turut

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            # ── Train phase ───────────────────────────────────────────────
            self.model.train()
            train_loss_accum = 0.0
            train_steps = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss_accum += loss.item()
                train_steps += 1

            avg_train_loss = train_loss_accum / max(train_steps, 1)

            # ── Validation phase ──────────────────────────────────────────
            val_loss, val_accuracy = self._evaluate_epoch(
                val_loader, criterion
            )

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 6),
            }
            history.append(epoch_record)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f, val_loss=%.4f, val_accuracy=%.4f",
                epoch,
                epochs,
                avg_train_loss,
                val_loss,
                val_accuracy,
            )

            # Callback
            if callback is not None:
                try:
                    callback(epoch, avg_train_loss, val_loss, val_accuracy)
                except Exception as exc:
                    logger.warning("Callback error di epoch %d: %s", epoch, exc)

            # ── Best model tracking ───────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Simpan checkpoint terbaik
                self.save_model(str(self.save_dir))
                logger.info(
                    "Model terbaik disimpan (epoch %d, val_loss=%.4f).",
                    epoch,
                    val_loss,
                )
            else:
                patience_counter += 1
                logger.info(
                    "Val loss tidak membaik (%d/%d).", patience_counter, patience_limit
                )

            # ── Early stopping ────────────────────────────────────────────
            if patience_counter >= patience_limit:
                logger.info(
                    "Early stopping di epoch %d (patience=%d tercapai).",
                    epoch,
                    patience_limit,
                )
                break

        training_time = time.time() - start_time

        result: dict[str, Any] = {
            "model_type": self.model_type,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "best_val_accuracy": round(best_val_accuracy, 6),
            "history": history,
            "model_path": str(self.save_dir),
            "training_time_seconds": round(training_time, 2),
        }

        logger.info(
            "Pelatihan selesai dalam %.1f detik. Best epoch=%d, "
            "val_loss=%.4f, val_accuracy=%.4f",
            training_time,
            best_epoch,
            best_val_loss,
            best_val_accuracy,
        )
        return result

    # ── Evaluation Helper ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate_epoch(
        self,
        data_loader: DataLoader,
        criterion: torch.nn.Module,
    ) -> tuple[float, float]:
        """Evaluasi model pada satu epoch (validasi).

        Returns:
            Tuple (avg_val_loss, accuracy).
        """
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        eval_steps = 0

        for batch in data_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            eval_steps += 1

            preds = torch.argmax(outputs.logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / max(eval_steps, 1)
        accuracy = total_correct / max(total_samples, 1)
        return avg_loss, accuracy

    # ── Model Persistence ─────────────────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Menyimpan model, tokenizer, dan pemetaan label ke direktori.

        Args:
            path: Direktori tujuan penyimpanan.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model dan tokenizer belum diinisialisasi. "
                "Jalankan train() terlebih dahulu."
            )

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        # Simpan label mapping sebagai JSON terpisah untuk kemudahan akses
        label_mapping = {
            "label2id": self.label2id,
            "id2label": {str(k): v for k, v in self.id2label.items()},
            "labels": list(self.label2id.keys()),
        }
        mapping_path = save_path / "label_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, ensure_ascii=False, indent=2)

        logger.info(
            "Model, tokenizer, dan label mapping disimpan ke '%s'.", save_path
        )
