"""
csv_export.py
-------------
Modul ekspor data analisis sentimen ke format CSV dan JSON.
Mendukung UTF-8 BOM encoding agar kompatibel dengan Microsoft Excel,
serta penanganan datetime, NaN, dan karakter khusus.
"""
from __future__ import annotations

import csv
import io
import json
import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Kolom ekspor default ──────────────────────────────────────────────────────
ANALYSIS_COLUMNS: list[str] = [
    "Tanggal", "Waktu", "X akun", "Konten", "Komentar", "Repost",
    "Likes", "Views", "Link",
    "clean_text",
    "sentiment", "confidence",
    "prob_positive", "prob_neutral", "prob_negative",
    "emotion", "sarcasm",
]


# ── Helper Internal ───────────────────────────────────────────────────────────

def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bersihkan DataFrame untuk ekspor:
    - Konversi datetime ke string ISO
    - Ganti NaN/inf dengan string kosong
    - Truncate teks sangat panjang
    """
    df = df.copy()

    for col in df.columns:
        # Konversi kolom datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Konversi timedelta
        if pd.api.types.is_timedelta64_dtype(df[col]):
            df[col] = df[col].astype(str)

    # Ganti NaN, inf, -inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna("")

    return df


def _select_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Pilih kolom yang tersedia dari daftar kolom target."""
    if columns is None:
        columns = ANALYSIS_COLUMNS

    available = [c for c in columns if c in df.columns]

    # Tambahkan kolom ekstra yang mungkin ada di df tapi bukan di daftar default
    extra = [c for c in df.columns if c not in available]
    all_cols = available + extra

    return df[all_cols]


def _escape_formula_injection(val: Any) -> Any:
    """
    Cegah CSV formula injection.
    Jika nilai string dimulai dengan =, +, -, @, prefix dengan apostrof.
    """
    if isinstance(val, str) and len(val) > 0 and val[0] in ("=", "+", "-", "@"):
        return f"'{val}"
    return val


# ── Public API ────────────────────────────────────────────────────────────────

def generate_csv_report(df: pd.DataFrame) -> str:
    """
    Generate CSV string dari DataFrame hasil analisis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame berisi data asli + kolom hasil analisis.

    Returns
    -------
    str
        CSV string (UTF-8, tanpa BOM).

    Raises
    ------
    ValueError
        Jika DataFrame kosong.
    """
    if df is None or df.empty:
        logger.warning("DataFrame kosong, mengembalikan CSV header saja.")
        return ",".join(ANALYSIS_COLUMNS) + "\n"

    logger.info("Generating CSV report untuk %d baris.", len(df))

    export_df = _select_columns(df)
    export_df = _sanitize_dataframe(export_df)

    # Terapkan formula injection protection
    export_df = export_df.map(_escape_formula_injection)

    output = io.StringIO()
    export_df.to_csv(output, index=False, quoting=csv.QUOTE_NONNUMERIC,
                     encoding="utf-8")
    csv_str = output.getvalue()

    logger.info("CSV report berhasil digenerate (%d karakter).", len(csv_str))
    return csv_str


def generate_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Generate CSV bytes dengan UTF-8 BOM encoding untuk kompatibilitas Excel.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame berisi data analisis.

    Returns
    -------
    bytes
        CSV content dengan UTF-8 BOM prefix.
    """
    csv_str = generate_csv_report(df)

    # UTF-8 BOM prefix agar Excel mengenali encoding dengan benar
    bom = b"\xef\xbb\xbf"
    csv_bytes = bom + csv_str.encode("utf-8")

    logger.info("CSV bytes berhasil digenerate (%d bytes).", len(csv_bytes))
    return csv_bytes


def export_to_json(df: pd.DataFrame) -> str:
    """
    Ekspor DataFrame ke JSON string (array of objects).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame berisi data analisis.

    Returns
    -------
    str
        JSON string dengan format yang rapi.

    Notes
    -----
    - Datetime dikonversi ke ISO 8601 string.
    - NaN dikonversi ke null.
    - Encoding UTF-8 untuk karakter Indonesia.
    """
    if df is None or df.empty:
        logger.warning("DataFrame kosong, mengembalikan JSON array kosong.")
        return "[]"

    logger.info("Generating JSON export untuk %d baris.", len(df))

    export_df = _select_columns(df)
    export_df = _sanitize_dataframe(export_df)

    # Konversi ke list of dicts
    records = export_df.to_dict(orient="records")

    # Custom JSON serializer untuk tipe data non-standard
    def _default_serializer(obj: Any) -> Any:
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return str(obj)

    json_str = json.dumps(
        records,
        ensure_ascii=False,  # izinkan karakter Unicode Indonesia
        indent=2,
        default=_default_serializer,
    )

    logger.info("JSON export berhasil digenerate (%d karakter).", len(json_str))
    return json_str


def export_to_json_bytes(df: pd.DataFrame) -> bytes:
    """
    Ekspor DataFrame ke JSON bytes (UTF-8).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame berisi data analisis.

    Returns
    -------
    bytes
        JSON content sebagai bytes UTF-8.
    """
    json_str = export_to_json(df)
    return json_str.encode("utf-8")
