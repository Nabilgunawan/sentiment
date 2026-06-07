"""
analytics/influencer_analyzer.py
---------------------------------
Modul analisis influencer untuk platform analisis sentimen media sosial Indonesia.

Fitur:
- Hitung influence score berdasarkan bobot post, engagement, dan views
- Identifikasi top-N influencer
- Temukan akun yang mendorong sentimen positif/negatif (sentiment leaders)
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bobot untuk perhitungan influence score
_WEIGHT_POST_COUNT: float = 0.2
_WEIGHT_ENGAGEMENT: float = 0.5
_WEIGHT_VIEWS: float = 0.3


class InfluencerAnalyzer:
    """
    Kelas untuk menganalisis influencer dari data sentimen media sosial.

    Influence score dihitung menggunakan formula berbobot:
        score = (norm_post * W_POST) + (norm_engagement * W_ENG) + (norm_views * W_VIEWS)

    di mana norm_* = nilai dinormalisasi ke [0, 1] menggunakan min-max scaling.
    """

    def __init__(
        self,
        weight_post: float = _WEIGHT_POST_COUNT,
        weight_engagement: float = _WEIGHT_ENGAGEMENT,
        weight_views: float = _WEIGHT_VIEWS,
    ) -> None:
        """
        Args:
            weight_post: Bobot untuk jumlah post (default 0.2).
            weight_engagement: Bobot untuk total engagement (default 0.5).
            weight_views: Bobot untuk total views (default 0.3).
        """
        self.weight_post = weight_post
        self.weight_engagement = weight_engagement
        self.weight_views = weight_views

    # ----------------------------------------------------------------
    # Utilitas internal
    # ----------------------------------------------------------------

    @staticmethod
    def _min_max_normalize(series: pd.Series) -> pd.Series:
        """Normalisasi min-max ke rentang [0, 1]."""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series(0.5, index=series.index)
        return (series - min_val) / (max_val - min_val)

    @staticmethod
    def _safe_get_sarcasm_ratio(group: pd.DataFrame) -> float:
        """Hitung rasio sarkasme dari grup post satu akun."""
        if "sarcasm" not in group.columns or group.empty:
            return 0.0

        sarcasm_col = group["sarcasm"].fillna(False)
        sarcastic_mask = sarcasm_col.apply(
            lambda v: v is True
            or (isinstance(v, str) and v.lower() in ("ya", "yes", "sarcastic", "true", "1"))
            or v == 1
        )
        total = len(group)
        return round(float(sarcastic_mask.sum() / total * 100), 1) if total else 0.0

    @staticmethod
    def _get_distribution(series: pd.Series) -> Dict[str, int]:
        """Hitung distribusi frekuensi dari Series."""
        if series.empty:
            return {}
        counts = series.value_counts().to_dict()
        return {str(k): int(v) for k, v in counts.items()}

    # ----------------------------------------------------------------
    # Analisis Utama
    # ----------------------------------------------------------------

    def analyze(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analisis lengkap semua influencer dalam dataset.

        Args:
            df: DataFrame lengkap hasil analisis sentimen.

        Returns:
            List dict per influencer, diurutkan berdasarkan influence_score:
                akun, total_post, total_engagement, avg_engagement,
                reach_score, dominant_sentiment, sentiment_distribution,
                dominant_emotion, emotion_distribution, sarcasm_ratio,
                influence_score
        """
        if df.empty:
            logger.info("DataFrame kosong, tidak ada influencer untuk dianalisis.")
            return []

        if "X akun" not in df.columns:
            logger.warning("Kolom 'X akun' tidak ditemukan.")
            return []

        score_col = (
            "engagement_score"
            if "engagement_score" in df.columns
            else ("engagement" if "engagement" in df.columns else None)
        )

        # ── Agregasi per akun ──
        records: List[Dict[str, Any]] = []
        for akun, group in df.groupby("X akun"):
            total_post = len(group)
            total_eng = float(group[score_col].sum()) if score_col else 0.0
            avg_eng = float(group[score_col].mean()) if score_col else 0.0
            total_views = float(group["Views"].sum()) if "Views" in group.columns else 0.0

            # Reach score = total views (proksi jangkauan)
            reach_score = total_views

            # Sentimen
            dominant_sentiment = "Neutral"
            sentiment_dist: Dict[str, int] = {}
            if "sentiment" in group.columns:
                sentiment_dist = self._get_distribution(group["sentiment"])
                sent_counts = group["sentiment"].value_counts()
                if not sent_counts.empty:
                    dominant_sentiment = str(sent_counts.index[0])

            # Emosi
            dominant_emotion = "Netral"
            emotion_dist: Dict[str, int] = {}
            if "emotion" in group.columns:
                emotion_dist = self._get_distribution(group["emotion"])
                emo_counts = group["emotion"].value_counts()
                if not emo_counts.empty:
                    dominant_emotion = str(emo_counts.index[0])

            # Sarkasme
            sarcasm_ratio = self._safe_get_sarcasm_ratio(group)

            records.append({
                "akun":                   str(akun),
                "total_post":             total_post,
                "total_engagement":       round(total_eng, 2),
                "avg_engagement":         round(avg_eng, 2),
                "reach_score":            round(reach_score, 2),
                "dominant_sentiment":     dominant_sentiment,
                "sentiment_distribution": sentiment_dist,
                "dominant_emotion":       dominant_emotion,
                "emotion_distribution":   emotion_dist,
                "sarcasm_ratio":          sarcasm_ratio,
                "influence_score":        0.0,  # placeholder, dihitung setelah normalisasi
            })

        if not records:
            return []

        # ── Hitung influence score dengan normalisasi ──
        temp_df = pd.DataFrame(records)
        norm_post = self._min_max_normalize(temp_df["total_post"].astype(float))
        norm_eng = self._min_max_normalize(temp_df["total_engagement"].astype(float))
        norm_views = self._min_max_normalize(temp_df["reach_score"].astype(float))

        influence_scores = (
            norm_post * self.weight_post
            + norm_eng * self.weight_engagement
            + norm_views * self.weight_views
        )

        # Skalakan ke 0-100
        max_score = influence_scores.max()
        if max_score > 0:
            influence_scores = (influence_scores / max_score * 100).round(2)
        else:
            influence_scores = influence_scores.round(2)

        for i, score in enumerate(influence_scores):
            records[i]["influence_score"] = float(score)

        # Urutkan berdasarkan influence score
        records.sort(key=lambda x: x["influence_score"], reverse=True)
        logger.info("Analisis selesai untuk %d influencer.", len(records))
        return records

    # ----------------------------------------------------------------
    # Top-N Influencer
    # ----------------------------------------------------------------

    def get_top_influencers(
        self,
        df: pd.DataFrame,
        n: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Ambil top-N influencer berdasarkan influence score.

        Args:
            df: DataFrame lengkap.
            n:  Jumlah influencer teratas (default 10).

        Returns:
            List dict top-N influencer.
        """
        all_influencers = self.analyze(df)
        result = all_influencers[:n]
        logger.info("Mengembalikan %d top influencer.", len(result))
        return result

    # ----------------------------------------------------------------
    # Sentiment Leaders
    # ----------------------------------------------------------------

    def get_sentiment_leaders(
        self,
        df: pd.DataFrame,
        n: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Temukan akun yang paling mendorong sentimen positif dan negatif.

        Kriteria:
        - Positive leaders: akun dengan persentase post positif tertinggi
          (minimal 3 post)
        - Negative leaders: akun dengan persentase post negatif tertinggi
          (minimal 3 post)

        Args:
            df: DataFrame lengkap.
            n:  Jumlah leader per kategori (default 5).

        Returns:
            dict: {
                "positive_leaders": [...],
                "negative_leaders": [...],
            }
        """
        if df.empty or "X akun" not in df.columns or "sentiment" not in df.columns:
            return {"positive_leaders": [], "negative_leaders": []}

        MIN_POSTS = 3
        score_col = (
            "engagement_score"
            if "engagement_score" in df.columns
            else ("engagement" if "engagement" in df.columns else None)
        )

        leaders_data: List[Dict[str, Any]] = []
        for akun, group in df.groupby("X akun"):
            total = len(group)
            if total < MIN_POSTS:
                continue

            pos_count = int((group["sentiment"] == "Positive").sum())
            neg_count = int((group["sentiment"] == "Negative").sum())
            pos_pct = round(pos_count / total * 100, 1)
            neg_pct = round(neg_count / total * 100, 1)
            total_eng = round(float(group[score_col].sum()), 2) if score_col else 0.0

            leaders_data.append({
                "akun":             str(akun),
                "total_post":       total,
                "positive_count":   pos_count,
                "negative_count":   neg_count,
                "positive_pct":     pos_pct,
                "negative_pct":     neg_pct,
                "total_engagement": total_eng,
            })

        if not leaders_data:
            return {"positive_leaders": [], "negative_leaders": []}

        # Positive leaders: sort by positive_pct descending
        positive_leaders = sorted(
            leaders_data,
            key=lambda x: (x["positive_pct"], x["total_engagement"]),
            reverse=True,
        )[:n]

        # Negative leaders: sort by negative_pct descending
        negative_leaders = sorted(
            leaders_data,
            key=lambda x: (x["negative_pct"], x["total_engagement"]),
            reverse=True,
        )[:n]

        logger.info(
            "Sentiment leaders: %d positive, %d negative.",
            len(positive_leaders), len(negative_leaders),
        )
        return {
            "positive_leaders": positive_leaders,
            "negative_leaders": negative_leaders,
        }
