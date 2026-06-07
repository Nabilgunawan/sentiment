"""
training/data_utils.py
----------------------
Utilitas penanganan data pelatihan untuk platform Analisis Sentimen Media Sosial
Indonesia.  Menyediakan fungsi-fungsi untuk memuat data, validasi label,
pembagian train/val, pembuatan Dataset PyTorch, dan perhitungan bobot kelas
untuk menangani ketidakseimbangan data.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

import config

logger = logging.getLogger(__name__)

# ─── Dataclass ────────────────────────────────────────────────────────────────


@dataclass
class DatasetStats:
    """Statistik ringkasan dataset pelatihan.

    Attributes:
        total: Jumlah total sampel.
        per_class_counts: Distribusi jumlah sampel per kelas.
        train_size: Jumlah sampel untuk pelatihan.
        val_size: Jumlah sampel untuk validasi.
    """

    total: int = 0
    per_class_counts: dict[str, int] = field(default_factory=dict)
    train_size: int = 0
    val_size: int = 0

    def __repr__(self) -> str:  # pragma: no cover
        lines = [
            f"DatasetStats(total={self.total}, train={self.train_size}, val={self.val_size})",
        ]
        for label, count in sorted(self.per_class_counts.items()):
            pct = (count / self.total * 100) if self.total else 0.0
            lines.append(f"  {label}: {count} ({pct:.1f}%)")
        return "\n".join(lines)


# ─── Dataset Class ────────────────────────────────────────────────────────────


class SentimentDataset(Dataset):
    """PyTorch Dataset untuk tugas klasifikasi teks sentimen / emosi / sarkasme.

    Melakukan tokenisasi teks dan encoding label secara *lazy* per-item
    sehingga memori lebih efisien untuk dataset besar.

    Args:
        texts: Daftar teks masukan.
        labels: Daftar label string yang sesuai.
        tokenizer: Tokenizer HuggingFace yang sudah dimuat.
        label2id: Pemetaan label string → integer.
        max_length: Panjang token maksimal (padding/truncation).
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        max_length: int = 128,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError(
                f"Jumlah teks ({len(texts)}) dan label ({len(labels)}) tidak sama."
            )
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_id = self.label2id.get(label)
        if label_id is None:
            raise KeyError(
                f"Label '{label}' tidak ditemukan di label2id. "
                f"Label yang tersedia: {list(self.label2id.keys())}"
            )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }


# ─── Data Loading ─────────────────────────────────────────────────────────────


def load_training_data(
    filepath: str,
    text_col: str,
    label_col: str,
) -> tuple[list[str], list[str]]:
    """Memuat data pelatihan dari file CSV atau XLSX.

    Args:
        filepath: Path ke file data (format .csv, .xlsx, atau .xls).
        text_col: Nama kolom yang berisi teks.
        label_col: Nama kolom yang berisi label.

    Returns:
        Tuple (texts, labels) berupa daftar string.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
        ValueError: Jika kolom tidak ditemukan atau data kosong.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    ext = path.suffix.lower()
    logger.info("Memuat data pelatihan dari %s (format: %s)", filepath, ext)

    try:
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(filepath)
        else:
            raise ValueError(
                f"Format file '{ext}' tidak didukung. Gunakan CSV atau XLSX."
            )
    except Exception as exc:
        logger.error("Gagal membaca file %s: %s", filepath, exc)
        raise

    # Validasi kolom
    missing_cols = [c for c in (text_col, label_col) if c not in df.columns]
    if missing_cols:
        available = ", ".join(df.columns.tolist())
        raise ValueError(
            f"Kolom tidak ditemukan: {missing_cols}. Kolom tersedia: {available}"
        )

    # Hapus baris dengan nilai kosong di kolom penting
    initial_len = len(df)
    df = df.dropna(subset=[text_col, label_col])
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning(
            "Menghapus %d baris dengan nilai kosong di kolom '%s' atau '%s'.",
            dropped,
            text_col,
            label_col,
        )

    if df.empty:
        raise ValueError("Dataset kosong setelah menghapus baris kosong.")

    texts: list[str] = df[text_col].astype(str).tolist()
    labels: list[str] = df[label_col].astype(str).str.strip().tolist()

    logger.info(
        "Data dimuat: %d sampel. Distribusi label: %s",
        len(texts),
        dict(Counter(labels)),
    )
    return texts, labels


# ─── Label Utilities ──────────────────────────────────────────────────────────


def validate_labels(
    labels: list[str],
    valid_labels: list[str],
) -> tuple[bool, list[str]]:
    """Memvalidasi bahwa semua label termasuk dalam daftar label yang valid.

    Args:
        labels: Daftar label yang akan divalidasi.
        valid_labels: Daftar label yang diperbolehkan.

    Returns:
        Tuple (is_valid, invalid_labels). ``is_valid`` bernilai True jika semua
        label valid.  ``invalid_labels`` berisi daftar label unik yang tidak
        valid (kosong jika semua valid).
    """
    valid_set = set(valid_labels)
    invalid = sorted({lbl for lbl in labels if lbl not in valid_set})
    is_valid = len(invalid) == 0

    if not is_valid:
        logger.warning(
            "Ditemukan %d label tidak valid: %s. Label yang diperbolehkan: %s",
            len(invalid),
            invalid,
            valid_labels,
        )

    return is_valid, invalid


def get_label_mapping(
    labels: list[str],
) -> tuple[dict[str, int], dict[int, str]]:
    """Membuat pemetaan label ↔ ID numerik dari daftar label unik.

    Args:
        labels: Daftar label (urutan dipertahankan sesuai kemunculan pertama,
                kemudian diurutkan secara alfanumerik agar deterministik).

    Returns:
        Tuple (label2id, id2label).
    """
    unique_labels = sorted(set(labels))
    label2id: dict[str, int] = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id2label: dict[int, str] = {idx: lbl for lbl, idx in label2id.items()}
    logger.debug("Label mapping dibuat: %s", label2id)
    return label2id, id2label


# ─── Data Splitting ───────────────────────────────────────────────────────────


def split_data(
    texts: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Membagi data menjadi set pelatihan dan validasi menggunakan stratified split.

    Args:
        texts: Daftar teks.
        labels: Daftar label.
        test_size: Proporsi data untuk validasi (default 0.2).
        random_state: Seed untuk reprodusibilitas.

    Returns:
        Tuple (train_texts, val_texts, train_labels, val_labels).
    """
    if len(texts) != len(labels):
        raise ValueError(
            f"Panjang texts ({len(texts)}) dan labels ({len(labels)}) harus sama."
        )

    # Stratified split jika memungkinkan, fallback ke random split
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            list(texts),
            list(labels),
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError:
        # Stratified split gagal (misalnya kelas terlalu sedikit), gunakan random split
        logger.warning(
            "Stratified split gagal, menggunakan random split biasa."
        )
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            list(texts),
            list(labels),
            test_size=test_size,
            random_state=random_state,
        )

    logger.info(
        "Data dibagi: train=%d, val=%d (test_size=%.2f)",
        len(train_texts),
        len(val_texts),
        test_size,
    )
    return train_texts, val_texts, train_labels, val_labels


# ─── Dataset Creation ─────────────────────────────────────────────────────────


def create_dataset(
    texts: list[str],
    labels: list[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    label2id: Optional[dict[str, int]] = None,
) -> SentimentDataset:
    """Membuat instance ``SentimentDataset`` dari teks dan label.

    Jika ``label2id`` tidak disediakan, pemetaan akan dihasilkan secara
    otomatis dari label yang diberikan.

    Args:
        texts: Daftar teks masukan.
        labels: Daftar label string.
        tokenizer: Tokenizer HuggingFace.
        max_length: Panjang token maksimal.
        label2id: Opsional, pemetaan label → ID.

    Returns:
        Instance ``SentimentDataset`` yang siap digunakan oleh DataLoader.
    """
    if label2id is None:
        label2id, _ = get_label_mapping(labels)

    dataset = SentimentDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )
    logger.debug(
        "Dataset dibuat: %d sampel, max_length=%d",
        len(dataset),
        max_length,
    )
    return dataset


# ─── Class Weight Computation ─────────────────────────────────────────────────


def get_class_weights(labels: list[str]) -> torch.Tensor:
    """Menghitung bobot kelas untuk menangani ketidakseimbangan data.

    Menggunakan ``sklearn.utils.class_weight.compute_class_weight`` dengan
    strategi ``'balanced'``.

    Args:
        labels: Daftar label string.

    Returns:
        ``torch.Tensor`` berisi bobot kelas, diurutkan sesuai indeks label
        (alfanumerik).
    """
    unique_labels = sorted(set(labels))
    labels_array = np.array(labels)

    try:
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array(unique_labels),
            y=labels_array,
        )
    except Exception as exc:
        logger.warning(
            "Gagal menghitung class weights: %s. Menggunakan bobot uniform.",
            exc,
        )
        weights = np.ones(len(unique_labels), dtype=np.float64)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(
        "Class weights (urutan %s): %s",
        unique_labels,
        weights_tensor.tolist(),
    )
    return weights_tensor


# ─── Stats Helper ─────────────────────────────────────────────────────────────


def compute_dataset_stats(
    labels: list[str],
    train_labels: list[str] | None = None,
    val_labels: list[str] | None = None,
) -> DatasetStats:
    """Menghitung statistik dataset.

    Args:
        labels: Semua label (sebelum split).
        train_labels: Label set pelatihan (opsional).
        val_labels: Label set validasi (opsional).

    Returns:
        Instance ``DatasetStats``.
    """
    counts = dict(Counter(labels))
    return DatasetStats(
        total=len(labels),
        per_class_counts=counts,
        train_size=len(train_labels) if train_labels else 0,
        val_size=len(val_labels) if val_labels else 0,
    )
