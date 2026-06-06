import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime


def calculate_engagement(df):
    if "Komentar" in df.columns and "Repost" in df.columns and "Likes" in df.columns:
        df["engagement"] = (
            df["Komentar"].fillna(0) + df["Repost"].fillna(0) + df["Likes"].fillna(0)
        )
    elif "Komentar" in df.columns and "Repost" in df.columns:
        df["engagement"] = df["Komentar"].fillna(0) + df["Repost"].fillna(0)
    elif "Komentar" in df.columns and "Likes" in df.columns:
        df["engagement"] = df["Komentar"].fillna(0) + df["Likes"].fillna(0)
    elif "Repost" in df.columns and "Likes" in df.columns:
        df["engagement"] = df["Repost"].fillna(0) + df["Likes"].fillna(0)
    elif "Likes" in df.columns:
        df["engagement"] = df["Likes"].fillna(0)
    else:
        df["engagement"] = 0
    return df


def aggregate_daily(df):
    df = df.copy()
    df["date_only"] = df["datetime"].dt.normalize()
    daily = df.groupby("date_only").agg(
        total_post=("Konten", "count"),
        positive=("sentiment", lambda x: (x == "Positive").sum()),
        neutral=("sentiment", lambda x: (x == "Neutral").sum()),
        negative=("sentiment", lambda x: (x == "Negative").sum()),
        total_engagement=("engagement", "sum"),
    ).reset_index()
    daily = daily.sort_values("date_only")
    if "Views" in df.columns:
        views_daily = df.groupby("date_only")["Views"].sum().reset_index()
        daily = daily.merge(views_daily, on="date_only", how="left")
        daily["Views"] = daily["Views"].fillna(0).astype(int)
    return daily


def aggregate_by_account(df):
    account = df.groupby("X akun").agg(
        total_post=("Konten", "count"),
        positive=("sentiment", lambda x: (x == "Positive").sum()),
        neutral=("sentiment", lambda x: (x == "Neutral").sum()),
        negative=("sentiment", lambda x: (x == "Negative").sum()),
        total_engagement=("engagement", "sum"),
    ).reset_index()
    account["engagement_rate"] = account["total_engagement"] / account["total_post"]
    account["engagement_rate"] = account["engagement_rate"].fillna(0).round(2)
    if "Views" in df.columns:
        views_acc = df.groupby("X akun")["Views"].sum().reset_index()
        account = account.merge(views_acc, on="X akun", how="left")
        account["Views"] = account["Views"].fillna(0).astype(int)
    account = account.sort_values("total_post", ascending=False)
    return account


def extract_top_keywords(df, text_column="clean_text", n=20):
    all_words = []
    for text in df[text_column].dropna():
        all_words.extend(text.split())
    counter = Counter(all_words)
    stopwords = {
        "yg", "nggak", "gak", "ga", "udah", "udh", "aja", "ajah",
        "bgt", "banget", "pake", "pkai", "dgn", "utk", "dr", "ttg",
    }
    filtered = [(w, c) for w, c in counter.most_common(n * 3) if w not in stopwords and len(w) > 1]
    return filtered[:n]


def get_kpis(df):
    total_post = len(df)
    unique_accounts = df["X akun"].nunique()
    positive = (df["sentiment"] == "Positive").sum()
    neutral = (df["sentiment"] == "Neutral").sum()
    negative = (df["sentiment"] == "Negative").sum()
    total_engagement = int(df["engagement"].sum())
    total_views = int(df["Views"].sum()) if "Views" in df.columns else 0
    avg_confidence = float(df["confidence"].mean())
    return {
        "total_post": total_post,
        "unique_accounts": unique_accounts,
        "positive": int(positive),
        "neutral": int(neutral),
        "negative": int(negative),
        "total_engagement": total_engagement,
        "total_views": total_views,
        "avg_confidence": round(avg_confidence, 3),
        "positive_pct": round(int(positive) / total_post * 100, 1) if total_post > 0 else 0,
        "neutral_pct": round(int(neutral) / total_post * 100, 1) if total_post > 0 else 0,
        "negative_pct": round(int(negative) / total_post * 100, 1) if total_post > 0 else 0,
    }
