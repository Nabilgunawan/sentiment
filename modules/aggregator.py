import pandas as pd
import numpy as np
from collections import Counter


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


def engagement_weighted_sentiment(df: pd.DataFrame) -> dict:
    """
    Hitung sentimen yang dibobotkan oleh engagement (engagement_weighted_sentiment).
    Post dengan engagement tinggi punya bobot lebih besar.

    Returns:
        dict: {sentiment: weighted_percentage}
    """
    if "engagement_score" not in df.columns:
        df = calculate_engagement_weighted(df)
    if "sentiment" not in df.columns:
        return {"Positive": 0, "Neutral": 0, "Negative": 0}

    total_weight = df["engagement_score"].sum()
    if total_weight == 0:
        pos = len(df[df["sentiment"] == "Positive"])
        neu = len(df[df["sentiment"] == "Neutral"])
        neg = len(df[df["sentiment"] == "Negative"])
        t = pos + neu + neg or 1
        return {"Positive": round(pos/t*100, 1), "Neutral": round(neu/t*100, 1), "Negative": round(neg/t*100, 1)}

    weighted = df.groupby("sentiment")["engagement_score"].sum()
    return {
        "Positive": round(float(weighted.get("Positive", 0) / total_weight * 100), 1),
        "Neutral":  round(float(weighted.get("Neutral", 0)  / total_weight * 100), 1),
        "Negative": round(float(weighted.get("Negative", 0) / total_weight * 100), 1),
    }


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi harian: jumlah post, sentimen, engagement."""
    df = df.copy()
    df["date_only"] = df["datetime"].dt.normalize()
    agg = df.groupby("date_only").agg(
        total_post=("Konten", "count"),
        positive=("sentiment", lambda x: (x == "Positive").sum()),
        neutral=("sentiment",  lambda x: (x == "Neutral").sum()),
        negative=("sentiment", lambda x: (x == "Negative").sum()),
        total_engagement=("engagement", "sum"),
    ).reset_index().sort_values("date_only")

    if "Views" in df.columns:
        views_d = df.groupby("date_only")["Views"].sum().reset_index()
        agg = agg.merge(views_d, on="date_only", how="left")
        agg["Views"] = agg["Views"].fillna(0).astype(int)

    return agg


def aggregate_by_account(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi per akun: post count, sentimen, engagement."""
    agg = df.groupby("X akun").agg(
        total_post=("Konten", "count"),
        positive=("sentiment", lambda x: (x == "Positive").sum()),
        neutral=("sentiment",  lambda x: (x == "Neutral").sum()),
        negative=("sentiment", lambda x: (x == "Negative").sum()),
        total_engagement=("engagement", "sum"),
    ).reset_index()
    agg["engagement_rate"] = (
        agg["total_engagement"] / agg["total_post"]
    ).fillna(0).round(2)

    if "Views" in df.columns:
        views_a = df.groupby("X akun")["Views"].sum().reset_index()
        agg = agg.merge(views_a, on="X akun", how="left")
        agg["Views"] = agg["Views"].fillna(0).astype(int)

    def dominant(row):
        return max(
            [("Positive", row["positive"]),
             ("Neutral",  row["neutral"]),
             ("Negative", row["negative"])],
            key=lambda x: x[1]
        )[0]
    agg["dominant_sentiment"] = agg.apply(dominant, axis=1)
    return agg.sort_values("total_post", ascending=False)


def aggregate_by_emotion(df: pd.DataFrame) -> dict:
    """
    Hitung distribusi 8 emosi dari kolom 'emotion' (hasil DeepSeek AI)
    atau dari deteksi lexicon.
    """
    EMOTIONS = ["Marah", "Takut", "Jijik", "Sedih",
                "Antisipasi", "Percaya", "Terkejut", "Senang"]

    if "emotion" in df.columns:
        counts = df["emotion"].value_counts().to_dict()
        total = len(df)
        result = {}
        for e in EMOTIONS:
            c = counts.get(e, 0)
            result[e] = {"count": int(c), "percentage": round(c / total * 100, 1) if total else 0}
        return result

    return {e: {"count": 0, "percentage": 0.0} for e in EMOTIONS}


_STOPWORDS = {
    "yg", "nggak", "gak", "ga", "udah", "udh", "aja", "ajah",
    "bgt", "banget", "pake", "dgn", "utk", "dr", "ttg", "jd",
    "lg", "lg", "juga", "yang", "dan", "di", "ke", "dari",
    "ini", "itu", "dengan", "untuk", "pada", "ada", "tidak",
    "akan", "sudah", "bisa", "dapat", "oleh", "dalam", "atau",
    "saya", "kamu", "dia", "kami", "kita", "mereka", "anda",
    "saat", "jika", "karena", "bahwa", "adalah", "bukan",
    "tapi", "tapi", "namun", "tp", "sdh", "skrg", "blm",
    "nya", "ku", "mu", "lah", "kah", "pun", "si", "sang",
    "the", "is", "in", "on", "at", "to", "of", "a", "an",
}


def extract_top_keywords(df: pd.DataFrame, text_column: str = "clean_text", n: int = 30) -> list:
    """Ambil n kata paling sering muncul (unigram)."""
    all_words = []
    for text in df[text_column].dropna():
        words = [w for w in str(text).lower().split()
                 if w not in _STOPWORDS and len(w) > 1]
        all_words.extend(words)
    counter = Counter(all_words)
    return counter.most_common(n)


def extract_bigrams(df: pd.DataFrame, text_column: str = "clean_text", n: int = 20) -> list:
    """Ambil n bigram (2-kata) paling sering muncul."""
    bigrams = []
    for text in df[text_column].dropna():
        words = [w for w in str(text).lower().split()
                 if w not in _STOPWORDS and len(w) > 1]
        for i in range(len(words) - 1):
            bigrams.append(f"{words[i]} {words[i+1]}")
    return Counter(bigrams).most_common(n)


def extract_trigrams(df: pd.DataFrame, text_column: str = "clean_text", n: int = 10) -> list:
    """Ambil n trigram (3-kata) paling sering muncul."""
    trigrams = []
    for text in df[text_column].dropna():
        words = [w for w in str(text).lower().split()
                 if w not in _STOPWORDS and len(w) > 1]
        for i in range(len(words) - 2):
            trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    return Counter(trigrams).most_common(n)


def detect_sentiment_spikes(daily_df: pd.DataFrame, threshold_std: float = 1.5) -> list:
    """
    Identifikasi hari dengan lonjakan sentimen negatif/positif signifikan
    (melebihi threshold_std standar deviasi dari rata-rata).

    Returns:
        List dict: [{date, type, magnitude, pct}]
    """
    spikes = []
    if len(daily_df) < 3:
        return spikes

    for col, label in [("negative", "Negatif"), ("positive", "Positif")]:
        if col not in daily_df.columns:
            continue
        series = daily_df[col].fillna(0).astype(float)
        mean = series.mean()
        std  = series.std()
        if std == 0:
            continue

        for _, row in daily_df.iterrows():
            val = row[col]
            z   = (val - mean) / std
            if z >= threshold_std:
                total = row.get("total_post", 1) or 1
                pct   = round(val / total * 100, 1)
                date_val = row["date_only"]
                if hasattr(date_val, "strftime"):
                    date_str = date_val.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_val)
                spikes.append({
                    "date":      date_str,
                    "type":      label,
                    "magnitude": round(float(z), 2),
                    "count":     int(val),
                    "pct":       pct,
                })

    spikes.sort(key=lambda x: x["magnitude"], reverse=True)
    return spikes[:5]


def get_top_viral_posts(df: pd.DataFrame, n: int = 10) -> list:
    """
    Ambil n post dengan engagement_score tertinggi.

    Returns:
        List dict dengan field utama post.
    """
    score_col = "engagement_score" if "engagement_score" in df.columns else "engagement"
    top = df.nlargest(n, score_col)

    result = []
    for _, row in top.iterrows():
        konten = str(row.get("Konten", ""))
        result.append({
            "akun":       str(row.get("X akun", "")),
            "konten":     konten,
            "konten_short": konten[:120] + "..." if len(konten) > 120 else konten,
            "sentiment":  str(row.get("sentiment", "Neutral")),
            "emotion":    str(row.get("emotion", "Netral")),
            "engagement": int(row.get("engagement", 0)),
            "engagement_score": float(row.get(score_col, 0)),
            "likes":      int(row.get("Likes", 0)),
            "repost":     int(row.get("Repost", 0)),
            "komentar":   int(row.get("Komentar", 0)),
            "views":      int(row.get("Views", 0)),
            "datetime":   str(row.get("datetime_str", "")),
        })
    return result


def get_sentiment_trend(kpis: dict) -> str:
    """Hitung trend sentimen dari perubahan hari pertama ke terakhir."""
    if kpis.get("positive_pct", 0) > kpis.get("negative_pct", 0) + 10:
        return "up"
    elif kpis.get("negative_pct", 0) > kpis.get("positive_pct", 0) + 10:
        return "down"
    return "stable"


def get_kpis(df: pd.DataFrame) -> dict:
    """Hitung semua Key Performance Indicators."""
    total     = len(df)
    unique_acc = df["X akun"].nunique()
    positive  = int((df["sentiment"] == "Positive").sum())
    neutral   = int((df["sentiment"] == "Neutral").sum())
    negative  = int((df["sentiment"] == "Negative").sum())
    total_eng = int(df["engagement"].sum())
    total_views = int(df["Views"].sum()) if "Views" in df.columns else 0
    avg_conf  = float(df["confidence"].mean()) if "confidence" in df.columns else 0.5

    pos_pct = round(positive / total * 100, 1) if total else 0
    neu_pct = round(neutral  / total * 100, 1) if total else 0
    neg_pct = round(negative / total * 100, 1) if total else 0

    return {
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
        "trend":            get_sentiment_trend({"positive_pct": pos_pct, "negative_pct": neg_pct}),
    }


def get_sentiment_index(kpis: dict) -> float:
    """
    Hitung Sentiment Index dari -100 (sangat negatif) sampai +100 (sangat positif).
    Formula: ((positive - negative) / total) * 100
    """
    total = kpis.get("total_post", 1) or 1
    pos   = kpis.get("positive", 0)
    neg   = kpis.get("negative", 0)
    return round((pos - neg) / total * 100, 1)
