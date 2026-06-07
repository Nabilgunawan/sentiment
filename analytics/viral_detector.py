"""
analytics/viral_detector.py
----------------------------
Modul deteksi konten viral untuk analisis sentimen media sosial Indonesia.

Fitur:
- Identifikasi top-N post dengan engagement tertinggi
- Deteksi thread/percakapan viral
- Statistik viral keseluruhan
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ====================================================================
# TOP VIRAL POSTS
# ====================================================================

def get_top_viral_posts(
    df: pd.DataFrame,
    n: int = 10,
) -> List[Dict[str, Any]]:
    """
    Ambil n post dengan engagement_score tertinggi.

    Dibandingkan versi lama, menambahkan field:
    sarcasm, emotion, prob_positive, prob_neutral, prob_negative.

    Args:
        df: DataFrame lengkap hasil analisis sentimen.
        n:  Jumlah post viral yang dikembalikan (default 10).

    Returns:
        List dict dengan field utama setiap post viral.
    """
    if df.empty:
        logger.info("DataFrame kosong, tidak ada viral post.")
        return []

    score_col = "engagement_score" if "engagement_score" in df.columns else "engagement"
    if score_col not in df.columns:
        logger.warning(
            "Kolom '%s' tidak ditemukan, mengembalikan list kosong.", score_col,
        )
        return []

    # Pastikan kolom numerik tidak NaN
    top = df.nlargest(min(n, len(df)), score_col)

    result: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        konten = str(row.get("Konten", ""))
        dt_val = row.get("datetime", "")
        dt_str = (
            dt_val.strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(dt_val, "strftime")
            else str(dt_val)
        )

        result.append({
            "akun":             str(row.get("X akun", "")),
            "konten":           konten,
            "konten_short":     konten[:120] + "..." if len(konten) > 120 else konten,
            "sentiment":        str(row.get("sentiment", "Neutral")),
            "emotion":          str(row.get("emotion", "Netral")),
            "sarcasm":          str(row.get("sarcasm", "")),
            "engagement":       int(row.get("engagement", 0)),
            "engagement_score": float(row.get(score_col, 0)),
            "likes":            int(row.get("Likes", 0)),
            "repost":           int(row.get("Repost", 0)),
            "komentar":         int(row.get("Komentar", 0)),
            "views":            int(row.get("Views", 0)),
            "datetime":         dt_str,
            "prob_positive":    round(float(row.get("prob_positive", 0)), 4),
            "prob_neutral":     round(float(row.get("prob_neutral", 0)), 4),
            "prob_negative":    round(float(row.get("prob_negative", 0)), 4),
        })

    logger.info("Mengembalikan %d viral post teratas.", len(result))
    return result


# ====================================================================
# VIRAL THREAD DETECTION
# ====================================================================

def detect_viral_threads(
    df: pd.DataFrame,
    engagement_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Deteksi thread/percakapan viral berdasarkan akun yang
    memiliki banyak post dengan engagement tinggi.

    Sebuah "thread viral" didefinisikan sebagai sekelompok post
    dari akun yang sama di mana rata-rata engagement melebihi threshold.

    Args:
        df: DataFrame lengkap hasil analisis.
        engagement_threshold: Ambang batas engagement untuk dianggap viral.
            Jika None, menggunakan mean + 1*std dari engagement_score.

    Returns:
        List dict per thread viral:
            akun, post_count, total_engagement, avg_engagement,
            dominant_sentiment, dominant_emotion, sarcasm_ratio,
            date_range, top_post_konten
    """
    if df.empty:
        logger.info("DataFrame kosong, tidak ada viral thread.")
        return []

    score_col = "engagement_score" if "engagement_score" in df.columns else "engagement"
    if score_col not in df.columns:
        logger.warning("Kolom '%s' tidak ditemukan.", score_col)
        return []

    # Hitung threshold otomatis jika tidak disediakan
    if engagement_threshold is None:
        mean_eng = df[score_col].mean()
        std_eng = df[score_col].std()
        engagement_threshold = mean_eng + std_eng
        logger.info(
            "Threshold engagement otomatis: %.2f (mean=%.2f, std=%.2f)",
            engagement_threshold, mean_eng, std_eng,
        )

    # Filter post dengan engagement di atas threshold
    viral_mask = df[score_col] >= engagement_threshold
    viral_posts = df[viral_mask]

    if viral_posts.empty:
        logger.info("Tidak ada post yang melampaui threshold engagement %.2f.", engagement_threshold)
        return []

    if "X akun" not in viral_posts.columns:
        logger.warning("Kolom 'X akun' tidak ditemukan.")
        return []

    threads: List[Dict[str, Any]] = []
    for akun, group in viral_posts.groupby("X akun"):
        post_count = len(group)
        total_eng = float(group[score_col].sum())
        avg_eng = float(group[score_col].mean())

        # Sentimen dominan
        dominant_sentiment = "Neutral"
        if "sentiment" in group.columns:
            sent_counts = group["sentiment"].value_counts()
            if not sent_counts.empty:
                dominant_sentiment = str(sent_counts.index[0])

        # Emosi dominan
        dominant_emotion = "Netral"
        if "emotion" in group.columns:
            emo_counts = group["emotion"].value_counts()
            if not emo_counts.empty:
                dominant_emotion = str(emo_counts.index[0])

        # Rasio sarkasme
        sarcasm_ratio = 0.0
        if "sarcasm" in group.columns:
            sarcasm_col = group["sarcasm"].fillna(False)
            sarcastic_mask = sarcasm_col.apply(
                lambda v: v is True
                or (isinstance(v, str) and v.lower() in ("ya", "yes", "sarcastic", "true", "1"))
                or v == 1
            )
            sarcasm_ratio = round(float(sarcastic_mask.sum() / post_count * 100), 1)

        # Rentang tanggal
        date_range = ""
        if "datetime" in group.columns:
            try:
                dates = pd.to_datetime(group["datetime"], errors="coerce").dropna()
                if not dates.empty:
                    date_range = f"{dates.min().strftime('%Y-%m-%d')} — {dates.max().strftime('%Y-%m-%d')}"
            except Exception:
                pass

        # Top post dalam thread
        top_post = group.nlargest(1, score_col).iloc[0]
        top_konten = str(top_post.get("Konten", ""))

        threads.append({
            "akun":               str(akun),
            "post_count":         post_count,
            "total_engagement":   round(total_eng, 2),
            "avg_engagement":     round(avg_eng, 2),
            "dominant_sentiment": dominant_sentiment,
            "dominant_emotion":   dominant_emotion,
            "sarcasm_ratio":      sarcasm_ratio,
            "date_range":         date_range,
            "top_post_konten":    top_konten[:200] + "..." if len(top_konten) > 200 else top_konten,
        })

    # Urutkan berdasarkan total engagement
    threads.sort(key=lambda x: x["total_engagement"], reverse=True)
    logger.info("Terdeteksi %d viral thread.", len(threads))
    return threads


# ====================================================================
# VIRAL STATISTICS
# ====================================================================

def get_viral_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Hitung statistik viral keseluruhan.

    Returns:
        dict:
            total_posts: jumlah total post
            avg_engagement: rata-rata engagement score
            median_engagement: median engagement score
            max_engagement: engagement tertinggi
            viral_count: jumlah post viral (> mean + 1.5*std)
            viral_ratio: persentase post viral
            viral_sentiment_dist: distribusi sentimen post viral
            viral_emotion_dist: distribusi emosi post viral
            top_viral_akun: akun dengan total viral engagement tertinggi
    """
    if df.empty:
        return {
            "total_posts":         0,
            "avg_engagement":      0.0,
            "median_engagement":   0.0,
            "max_engagement":      0.0,
            "viral_count":         0,
            "viral_ratio":         0.0,
            "viral_sentiment_dist": {},
            "viral_emotion_dist":  {},
            "top_viral_akun":      "",
        }

    score_col = "engagement_score" if "engagement_score" in df.columns else "engagement"
    if score_col not in df.columns:
        logger.warning("Kolom engagement tidak ditemukan untuk viral stats.")
        return {
            "total_posts":         len(df),
            "avg_engagement":      0.0,
            "median_engagement":   0.0,
            "max_engagement":      0.0,
            "viral_count":         0,
            "viral_ratio":         0.0,
            "viral_sentiment_dist": {},
            "viral_emotion_dist":  {},
            "top_viral_akun":      "",
        }

    scores = df[score_col].fillna(0)
    total = len(df)
    avg_eng = float(scores.mean())
    median_eng = float(scores.median())
    max_eng = float(scores.max())

    # Viral threshold: mean + 1.5 * std
    std_eng = float(scores.std()) if len(scores) > 1 else 0.0
    viral_threshold = avg_eng + 1.5 * std_eng
    viral_mask = scores >= viral_threshold
    viral_count = int(viral_mask.sum())
    viral_ratio = round(viral_count / total * 100, 1) if total else 0.0

    # Distribusi sentimen post viral
    viral_sentiment_dist: Dict[str, int] = {}
    if "sentiment" in df.columns and viral_count > 0:
        viral_sentiment_dist = (
            df.loc[viral_mask, "sentiment"]
            .value_counts()
            .to_dict()
        )
        viral_sentiment_dist = {str(k): int(v) for k, v in viral_sentiment_dist.items()}

    # Distribusi emosi post viral
    viral_emotion_dist: Dict[str, int] = {}
    if "emotion" in df.columns and viral_count > 0:
        viral_emotion_dist = (
            df.loc[viral_mask, "emotion"]
            .value_counts()
            .to_dict()
        )
        viral_emotion_dist = {str(k): int(v) for k, v in viral_emotion_dist.items()}

    # Akun viral teratas
    top_viral_akun = ""
    if "X akun" in df.columns and viral_count > 0:
        akun_eng = (
            df.loc[viral_mask]
            .groupby("X akun")[score_col]
            .sum()
            .sort_values(ascending=False)
        )
        if not akun_eng.empty:
            top_viral_akun = str(akun_eng.index[0])

    return {
        "total_posts":          total,
        "avg_engagement":       round(avg_eng, 2),
        "median_engagement":    round(median_eng, 2),
        "max_engagement":       round(max_eng, 2),
        "viral_count":          viral_count,
        "viral_ratio":          viral_ratio,
        "viral_sentiment_dist": viral_sentiment_dist,
        "viral_emotion_dist":   viral_emotion_dist,
        "top_viral_akun":       top_viral_akun,
    }
