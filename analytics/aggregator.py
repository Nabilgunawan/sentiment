"""
analytics/aggregator.py
-----------------------
Modul agregasi data sentimen media sosial Indonesia.

Migrasi lengkap dari modules/aggregator.py dengan penambahan fitur:
- Agregasi sarkasme (aggregate_by_sarcasm)
- Agregasi mingguan & bulanan (aggregate_weekly, aggregate_monthly)
- Tren emosi & sarkasme harian (aggregate_emotion_daily, aggregate_sarcasm_daily)
- KPI yang diperkaya (sarcasm_count, sarcasm_pct, avg_emotion)

Kolom DataFrame yang diharapkan:
    Konten, X akun, datetime, sentiment, emotion, sarcasm, confidence,
    engagement, engagement_score, Komentar, Repost, Likes, Views,
    clean_text, prob_positive, prob_neutral, prob_negative
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# Konstanta
# ────────────────────────────────────────────────────────────────────
EMOTIONS_7: List[str] = [
    "Marah", "Senang", "Sedih", "Takut", "Jijik", "Terkejut", "Netral",
]

EMOTIONS_8: List[str] = [
    "Marah", "Takut", "Jijik", "Sedih",
    "Antisipasi", "Percaya", "Terkejut", "Senang",
]

_STOPWORDS: set = {
    # ── Bahasa Indonesia informal ──
    "yg", "nggak", "gak", "ga", "udah", "udh", "aja", "ajah",
    "bgt", "banget", "pake", "dgn", "utk", "dr", "ttg", "jd",
    "lg", "juga", "yang", "dan", "di", "ke", "dari",
    "ini", "itu", "dengan", "untuk", "pada", "ada", "tidak",
    "akan", "sudah", "bisa", "dapat", "oleh", "dalam", "atau",
    "saya", "kamu", "dia", "kami", "kita", "mereka", "anda",
    "saat", "jika", "karena", "bahwa", "adalah", "bukan",
    "tapi", "namun", "tp", "sdh", "skrg", "blm",
    "nya", "ku", "mu", "lah", "kah", "pun", "si", "sang",
    # ── Bahasa Inggris umum ──
    "the", "is", "in", "on", "at", "to", "of", "a", "an",
}


# ====================================================================
# ENGAGEMENT
# ====================================================================

def calculate_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung engagement mentah: Komentar + Repost + Likes."""
    cols = ["Komentar", "Repost", "Likes"]
    present = [c for c in cols if c in df.columns]
    if present:
        df["engagement"] = sum(df[c].fillna(0) for c in present)
    else:
        df["engagement"] = 0
    return df


def calculate_engagement_weighted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung engagement score yang dibobotkan:
    score = Likes + Repost*2 + Komentar*1.5 + Views*0.01
    """
    likes    = df["Likes"].fillna(0)    if "Likes"    in df.columns else 0
    repost   = df["Repost"].fillna(0)   if "Repost"   in df.columns else 0
    komentar = df["Komentar"].fillna(0) if "Komentar" in df.columns else 0
    views    = df["Views"].fillna(0)    if "Views"    in df.columns else 0

    df["engagement_score"] = (
        likes + repost * 2 + komentar * 1.5 + views * 0.01
    ).round(2)
    return df


def engagement_weighted_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    """
    Hitung sentimen yang dibobotkan oleh engagement.
    Post dengan engagement tinggi punya bobot lebih besar.

    Returns:
        dict: {sentiment: weighted_percentage}
    """
    if "engagement_score" not in df.columns:
        df = calculate_engagement_weighted(df)
    if "sentiment" not in df.columns:
        return {"Positive": 0.0, "Neutral": 0.0, "Negative": 0.0}

    total_weight = df["engagement_score"].sum()
    if total_weight == 0:
        pos = len(df[df["sentiment"] == "Positive"])
        neu = len(df[df["sentiment"] == "Neutral"])
        neg = len(df[df["sentiment"] == "Negative"])
        t = pos + neu + neg or 1
        return {
            "Positive": round(pos / t * 100, 1),
            "Neutral":  round(neu / t * 100, 1),
            "Negative": round(neg / t * 100, 1),
        }

    weighted = df.groupby("sentiment")["engagement_score"].sum()
    return {
        "Positive": round(float(weighted.get("Positive", 0) / total_weight * 100), 1),
        "Neutral":  round(float(weighted.get("Neutral",  0) / total_weight * 100), 1),
        "Negative": round(float(weighted.get("Negative", 0) / total_weight * 100), 1),
    }


# ====================================================================
# AGREGASI HARIAN / MINGGUAN / BULANAN
# ====================================================================

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan kolom 'datetime' bertipe datetime."""
    if "datetime" not in df.columns:
        logger.warning("Kolom 'datetime' tidak ditemukan, agregasi waktu tidak mungkin.")
        return df
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi harian: jumlah post, sentimen, engagement."""
    if df.empty:
        logger.info("DataFrame kosong, mengembalikan DataFrame harian kosong.")
        return pd.DataFrame(columns=[
            "date_only", "total_post", "positive", "neutral",
            "negative", "total_engagement",
        ])

    df = _ensure_datetime(df.copy())
    df["date_only"] = df["datetime"].dt.normalize()

    agg = df.groupby("date_only").agg(
        total_post=("Konten", "count"),
        positive=("sentiment",  lambda x: (x == "Positive").sum()),
        neutral=("sentiment",   lambda x: (x == "Neutral").sum()),
        negative=("sentiment",  lambda x: (x == "Negative").sum()),
        total_engagement=("engagement", "sum"),
    ).reset_index().sort_values("date_only")

    if "Views" in df.columns:
        views_d = df.groupby("date_only")["Views"].sum().reset_index()
        agg = agg.merge(views_d, on="date_only", how="left")
        agg["Views"] = agg["Views"].fillna(0).astype(int)

    return agg


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregasi mingguan: jumlah post, sentimen, engagement.
    Kolom week_start = tanggal Senin awal minggu.
    """
    if df.empty:
        logger.info("DataFrame kosong, mengembalikan DataFrame mingguan kosong.")
        return pd.DataFrame(columns=[
            "week_start", "total_post", "positive", "neutral",
            "negative", "total_engagement",
        ])

    df = _ensure_datetime(df.copy())
    df["week_start"] = df["datetime"].dt.to_period("W").apply(lambda p: p.start_time)

    agg = df.groupby("week_start").agg(
        total_post=("Konten", "count"),
        positive=("sentiment",  lambda x: (x == "Positive").sum()),
        neutral=("sentiment",   lambda x: (x == "Neutral").sum()),
        negative=("sentiment",  lambda x: (x == "Negative").sum()),
        total_engagement=("engagement", "sum"),
    ).reset_index().sort_values("week_start")

    if "Views" in df.columns:
        views_w = df.groupby("week_start")["Views"].sum().reset_index()
        agg = agg.merge(views_w, on="week_start", how="left")
        agg["Views"] = agg["Views"].fillna(0).astype(int)

    return agg


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agregasi bulanan: jumlah post, sentimen, engagement.
    Kolom month = tanggal awal bulan (YYYY-MM-01).
    """
    if df.empty:
        logger.info("DataFrame kosong, mengembalikan DataFrame bulanan kosong.")
        return pd.DataFrame(columns=[
            "month", "total_post", "positive", "neutral",
            "negative", "total_engagement",
        ])

    df = _ensure_datetime(df.copy())
    df["month"] = df["datetime"].dt.to_period("M").apply(lambda p: p.start_time)

    agg = df.groupby("month").agg(
        total_post=("Konten", "count"),
        positive=("sentiment",  lambda x: (x == "Positive").sum()),
        neutral=("sentiment",   lambda x: (x == "Neutral").sum()),
        negative=("sentiment",  lambda x: (x == "Negative").sum()),
        total_engagement=("engagement", "sum"),
    ).reset_index().sort_values("month")

    if "Views" in df.columns:
        views_m = df.groupby("month")["Views"].sum().reset_index()
        agg = agg.merge(views_m, on="month", how="left")
        agg["Views"] = agg["Views"].fillna(0).astype(int)

    return agg


# ====================================================================
# AGREGASI PER AKUN
# ====================================================================

def aggregate_by_account(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi per akun: post count, sentimen, engagement."""
    if df.empty:
        logger.info("DataFrame kosong, mengembalikan DataFrame akun kosong.")
        return pd.DataFrame(columns=[
            "X akun", "total_post", "positive", "neutral",
            "negative", "total_engagement", "engagement_rate",
            "dominant_sentiment",
        ])

    agg = df.groupby("X akun").agg(
        total_post=("Konten", "count"),
        positive=("sentiment",  lambda x: (x == "Positive").sum()),
        neutral=("sentiment",   lambda x: (x == "Neutral").sum()),
        negative=("sentiment",  lambda x: (x == "Negative").sum()),
        total_engagement=("engagement", "sum"),
    ).reset_index()

    agg["engagement_rate"] = (
        agg["total_engagement"] / agg["total_post"]
    ).fillna(0).round(2)

    if "Views" in df.columns:
        views_a = df.groupby("X akun")["Views"].sum().reset_index()
        agg = agg.merge(views_a, on="X akun", how="left")
        agg["Views"] = agg["Views"].fillna(0).astype(int)

    def dominant(row: pd.Series) -> str:
        return max(
            [("Positive", row["positive"]),
             ("Neutral",  row["neutral"]),
             ("Negative", row["negative"])],
            key=lambda x: x[1],
        )[0]

    agg["dominant_sentiment"] = agg.apply(dominant, axis=1)
    return agg.sort_values("total_post", ascending=False)


# ====================================================================
# AGREGASI EMOSI
# ====================================================================

def aggregate_by_emotion(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Hitung distribusi emosi dari kolom 'emotion'.
    Mendukung 7 emosi utama: Marah, Senang, Sedih, Takut, Jijik, Terkejut, Netral.
    Juga menampilkan emosi tambahan dari EMOTIONS_8 jika ditemukan.
    """
    # Gabungkan semua label emosi yang dikenali
    all_emotions = list(dict.fromkeys(EMOTIONS_7 + EMOTIONS_8))

    if "emotion" not in df.columns or df.empty:
        return {e: {"count": 0, "percentage": 0.0} for e in all_emotions}

    counts = df["emotion"].value_counts().to_dict()
    total = len(df)
    result: Dict[str, Dict[str, Any]] = {}

    for e in all_emotions:
        c = counts.get(e, 0)
        result[e] = {
            "count": int(c),
            "percentage": round(c / total * 100, 1) if total else 0.0,
        }

    # Tangkap emosi yang tidak ada di daftar standar
    for e, c in counts.items():
        if e not in result:
            result[e] = {
                "count": int(c),
                "percentage": round(c / total * 100, 1) if total else 0.0,
            }

    return result


def aggregate_emotion_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tren emosi harian — setiap baris adalah satu tanggal,
    kolom = jumlah per emosi.
    """
    if df.empty or "emotion" not in df.columns:
        logger.info("Data kosong atau kolom 'emotion' tidak ada.")
        return pd.DataFrame()

    df = _ensure_datetime(df.copy())
    df["date_only"] = df["datetime"].dt.normalize()

    pivot = (
        df.groupby(["date_only", "emotion"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("date_only")
    )
    return pivot


# ====================================================================
# AGREGASI SARKASME
# ====================================================================

def aggregate_by_sarcasm(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Hitung distribusi sarkasme (sarcastic vs non-sarcastic).

    Returns:
        dict: {
            "sarcastic":     {"count": int, "percentage": float},
            "non_sarcastic": {"count": int, "percentage": float},
        }
    """
    if "sarcasm" not in df.columns or df.empty:
        return {
            "sarcastic":     {"count": 0, "percentage": 0.0},
            "non_sarcastic": {"count": 0, "percentage": 0.0},
        }

    total = len(df)
    sarcasm_col = df["sarcasm"].fillna(False)

    # Normalisasi: bisa bool, string 'Ya'/'Sarcastic', dsb.
    sarcastic_mask = sarcasm_col.apply(
        lambda v: v is True
        or (isinstance(v, str) and v.lower() in ("ya", "yes", "sarcastic", "true", "1"))
        or v == 1
    )
    sarcastic_count = int(sarcastic_mask.sum())
    non_sarcastic_count = total - sarcastic_count

    return {
        "sarcastic": {
            "count": sarcastic_count,
            "percentage": round(sarcastic_count / total * 100, 1) if total else 0.0,
        },
        "non_sarcastic": {
            "count": non_sarcastic_count,
            "percentage": round(non_sarcastic_count / total * 100, 1) if total else 0.0,
        },
    }


def aggregate_sarcasm_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tren sarkasme harian — kolom: date_only, sarcastic, non_sarcastic.
    """
    if df.empty or "sarcasm" not in df.columns:
        logger.info("Data kosong atau kolom 'sarcasm' tidak ada.")
        return pd.DataFrame(columns=["date_only", "sarcastic", "non_sarcastic"])

    df = _ensure_datetime(df.copy())
    df["date_only"] = df["datetime"].dt.normalize()

    sarcasm_col = df["sarcasm"].fillna(False)
    df["_is_sarcastic"] = sarcasm_col.apply(
        lambda v: v is True
        or (isinstance(v, str) and v.lower() in ("ya", "yes", "sarcastic", "true", "1"))
        or v == 1
    )

    agg = df.groupby("date_only").agg(
        sarcastic=("_is_sarcastic", "sum"),
        total=("_is_sarcastic", "count"),
    ).reset_index()
    agg["non_sarcastic"] = agg["total"] - agg["sarcastic"]
    agg["sarcastic"] = agg["sarcastic"].astype(int)
    agg["non_sarcastic"] = agg["non_sarcastic"].astype(int)
    return agg[["date_only", "sarcastic", "non_sarcastic"]].sort_values("date_only")


# ====================================================================
# KEYWORD EXTRACTION
# ====================================================================

def extract_top_keywords(
    df: pd.DataFrame,
    text_column: str = "clean_text",
    n: int = 30,
) -> List[Tuple[str, int]]:
    """Ambil n kata paling sering muncul (unigram)."""
    if text_column not in df.columns:
        logger.warning("Kolom '%s' tidak ditemukan untuk keyword extraction.", text_column)
        return []

    all_words: List[str] = []
    for text in df[text_column].dropna():
        words = [
            w for w in str(text).lower().split()
            if w not in _STOPWORDS and len(w) > 1
        ]
        all_words.extend(words)
    return Counter(all_words).most_common(n)


def extract_bigrams(
    df: pd.DataFrame,
    text_column: str = "clean_text",
    n: int = 20,
) -> List[Tuple[str, int]]:
    """Ambil n bigram (2-kata) paling sering muncul."""
    if text_column not in df.columns:
        logger.warning("Kolom '%s' tidak ditemukan untuk bigram extraction.", text_column)
        return []

    bigrams: List[str] = []
    for text in df[text_column].dropna():
        words = [
            w for w in str(text).lower().split()
            if w not in _STOPWORDS and len(w) > 1
        ]
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} {words[i+1]}")
    return Counter(bigrams).most_common(n)


def extract_trigrams(
    df: pd.DataFrame,
    text_column: str = "clean_text",
    n: int = 10,
) -> List[Tuple[str, int]]:
    """Ambil n trigram (3-kata) paling sering muncul."""
    if text_column not in df.columns:
        logger.warning("Kolom '%s' tidak ditemukan untuk trigram extraction.", text_column)
        return []

    trigrams: List[str] = []
    for text in df[text_column].dropna():
        words = [
            w for w in str(text).lower().split()
            if w not in _STOPWORDS and len(w) > 1
        ]
        for i in range(len(words) - 2):
            trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    return Counter(trigrams).most_common(n)


# ====================================================================
# SPIKE DETECTION (backward-compat wrapper)
# ====================================================================

def detect_sentiment_spikes(
    daily_df: pd.DataFrame,
    threshold_std: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Identifikasi hari dengan lonjakan sentimen negatif/positif signifikan
    (melebihi threshold_std standar deviasi dari rata-rata).

    Returns:
        List dict: [{date, type, magnitude, count, pct}]
    """
    spikes: List[Dict[str, Any]] = []
    if len(daily_df) < 3:
        return spikes

    for col, label in [("negative", "Negatif"), ("positive", "Positif")]:
        if col not in daily_df.columns:
            continue
        series = daily_df[col].fillna(0).astype(float)
        mean = series.mean()
        std = series.std()
        if std == 0:
            continue

        for _, row in daily_df.iterrows():
            val = row[col]
            z = (val - mean) / std
            if z >= threshold_std:
                total = row.get("total_post", 1) or 1
                pct = round(val / total * 100, 1)
                date_val = row["date_only"]
                date_str = (
                    date_val.strftime("%Y-%m-%d")
                    if hasattr(date_val, "strftime")
                    else str(date_val)
                )
                spikes.append({
                    "date":      date_str,
                    "type":      label,
                    "magnitude": round(float(z), 2),
                    "count":     int(val),
                    "pct":       pct,
                })

    spikes.sort(key=lambda x: x["magnitude"], reverse=True)
    return spikes[:5]


# ====================================================================
# VIRAL POSTS (backward-compat wrapper)
# ====================================================================

def get_top_viral_posts(df: pd.DataFrame, n: int = 10) -> List[Dict[str, Any]]:
    """
    Ambil n post dengan engagement_score tertinggi.

    Returns:
        List dict dengan field utama post.
    """
    if df.empty:
        return []

    score_col = "engagement_score" if "engagement_score" in df.columns else "engagement"
    if score_col not in df.columns:
        logger.warning("Kolom '%s' tidak ditemukan.", score_col)
        return []

    top = df.nlargest(n, score_col)

    result: List[Dict[str, Any]] = []
    for _, row in top.iterrows():
        konten = str(row.get("Konten", ""))
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
            "datetime":         str(row.get("datetime", "")),
            "prob_positive":    float(row.get("prob_positive", 0)),
            "prob_neutral":     float(row.get("prob_neutral", 0)),
            "prob_negative":    float(row.get("prob_negative", 0)),
        })
    return result


# ====================================================================
# SENTIMENT TREND & KPIs
# ====================================================================

def get_sentiment_trend(kpis: Dict[str, Any]) -> str:
    """Hitung trend sentimen dari perbandingan positif vs negatif."""
    if kpis.get("positive_pct", 0) > kpis.get("negative_pct", 0) + 10:
        return "up"
    elif kpis.get("negative_pct", 0) > kpis.get("positive_pct", 0) + 10:
        return "down"
    return "stable"


def get_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Hitung semua Key Performance Indicators.

    KPI yang dikembalikan:
        - total_post, unique_accounts
        - positive, neutral, negative (count & pct)
        - total_engagement, total_views, avg_confidence
        - sarcasm_count, sarcasm_pct
        - avg_emotion (emosi dominan rata-rata)
        - trend (up/down/stable)
    """
    if df.empty:
        return {
            "total_post": 0, "unique_accounts": 0,
            "positive": 0, "neutral": 0, "negative": 0,
            "total_engagement": 0, "total_views": 0,
            "avg_confidence": 0.0,
            "positive_pct": 0.0, "neutral_pct": 0.0, "negative_pct": 0.0,
            "sarcasm_count": 0, "sarcasm_pct": 0.0,
            "avg_emotion": "Netral",
            "trend": "stable",
        }

    total = len(df)
    unique_acc = df["X akun"].nunique() if "X akun" in df.columns else 0

    positive = int((df["sentiment"] == "Positive").sum()) if "sentiment" in df.columns else 0
    neutral  = int((df["sentiment"] == "Neutral").sum())  if "sentiment" in df.columns else 0
    negative = int((df["sentiment"] == "Negative").sum()) if "sentiment" in df.columns else 0

    total_eng   = int(df["engagement"].sum()) if "engagement" in df.columns else 0
    total_views = int(df["Views"].sum())      if "Views"      in df.columns else 0
    avg_conf    = float(df["confidence"].mean()) if "confidence" in df.columns else 0.5

    pos_pct = round(positive / total * 100, 1) if total else 0.0
    neu_pct = round(neutral  / total * 100, 1) if total else 0.0
    neg_pct = round(negative / total * 100, 1) if total else 0.0

    # ── Sarkasme ──
    sarcasm_info = aggregate_by_sarcasm(df)
    sarcasm_count = sarcasm_info["sarcastic"]["count"]
    sarcasm_pct   = sarcasm_info["sarcastic"]["percentage"]

    # ── Emosi dominan rata-rata ──
    avg_emotion = "Netral"
    if "emotion" in df.columns:
        emotion_counts = df["emotion"].value_counts()
        if not emotion_counts.empty:
            avg_emotion = str(emotion_counts.index[0])

    kpis: Dict[str, Any] = {
        "total_post":       total,
        "unique_accounts":  unique_acc,
        "positive":         positive,
        "neutral":          neutral,
        "negative":         negative,
        "total_engagement": total_eng,
        "total_views":      total_views,
        "avg_confidence":   round(avg_conf, 3),
        "positive_pct":     pos_pct,
        "neutral_pct":      neu_pct,
        "negative_pct":     neg_pct,
        "sarcasm_count":    sarcasm_count,
        "sarcasm_pct":      sarcasm_pct,
        "avg_emotion":      avg_emotion,
        "trend":            "stable",  # placeholder, dihitung di bawah
    }
    kpis["trend"] = get_sentiment_trend(kpis)
    return kpis


def get_sentiment_index(kpis: Dict[str, Any]) -> float:
    """
    Hitung Sentiment Index dari -100 (sangat negatif) sampai +100 (sangat positif).
    Formula: ((positive - negative) / total) * 100
    """
    total = kpis.get("total_post", 1) or 1
    pos   = kpis.get("positive", 0)
    neg   = kpis.get("negative", 0)
    return round((pos - neg) / total * 100, 1)
