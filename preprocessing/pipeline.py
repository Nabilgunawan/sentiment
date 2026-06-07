"""
preprocessing/pipeline.py
--------------------------
Pipeline preprocessing lengkap untuk model Transformer.
Menggabungkan cleaning dan normalisasi tanpa stemming/stopword removal.

Alur:
1. clean_text() → hapus URL, mention, hashtag symbol, emoji, HTML
2. normalize_text() → normalisasi slang, singkatan, typo
3. Return teks bersih siap untuk tokenizer Transformer
"""
import logging
import pandas as pd
from typing import Optional

from .cleaner import clean_text, clean_batch
from .normalizer import normalize_text, normalize_batch

logger = logging.getLogger(__name__)


def preprocess_text(text: str) -> str:
    """
    Pipeline preprocessing lengkap untuk satu teks.

    Args:
        text: Teks mentah dari media sosial

    Returns:
        Teks yang sudah di-preprocess, siap untuk tokenizer Transformer
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: Cleaning
    cleaned = clean_text(text)

    # Step 2: Normalisasi bahasa Indonesia
    normalized = normalize_text(cleaned)

    return normalized


def preprocess_batch(texts: list[str]) -> list[str]:
    """
    Pipeline preprocessing untuk batch teks.

    Args:
        texts: List teks mentah

    Returns:
        List teks yang sudah di-preprocess
    """
    # Step 1: Batch cleaning
    cleaned = clean_batch(texts)

    # Step 2: Batch normalisasi
    normalized = normalize_batch(cleaned)

    return normalized


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "Konten",
    output_column: str = "clean_text",
) -> pd.DataFrame:
    """
    Preprocess seluruh DataFrame — tambahkan kolom clean_text.

    Args:
        df: DataFrame dengan kolom teks
        text_column: Nama kolom yang berisi teks mentah
        output_column: Nama kolom output untuk teks bersih

    Returns:
        DataFrame dengan kolom baru 'clean_text'
    """
    df = df.copy()

    if text_column not in df.columns:
        logger.warning(f"Kolom '{text_column}' tidak ditemukan di DataFrame")
        df[output_column] = ""
        return df

    texts = df[text_column].fillna("").astype(str).tolist()
    preprocessed = preprocess_batch(texts)
    df[output_column] = preprocessed

    # Log stats
    total = len(texts)
    empty = sum(1 for t in preprocessed if not t.strip())
    logger.info(
        f"Preprocessing selesai: {total} teks diproses, "
        f"{empty} teks kosong setelah preprocessing"
    )

    return df
