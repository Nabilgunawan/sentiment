"""
analytics/spike_detector.py
----------------------------
Modul deteksi lonjakan (spike) untuk analisis sentimen media sosial Indonesia.

Mendeteksi anomali pada:
- Sentimen (positif/negatif) harian
- Emosi harian
- Engagement harian

Menggunakan z-score (standar deviasi dari rata-rata) untuk identifikasi spike.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpikeDetector:
    """
    Kelas untuk mendeteksi lonjakan (spike) pada data sentimen,
    emosi, dan engagement media sosial.

    Menggunakan metode z-score: nilai dianggap spike jika
    z-score >= threshold_std (default: 2.0).
    """

    def __init__(self, default_threshold: float = 2.0) -> None:
        """
        Args:
            default_threshold: Threshold z-score default untuk deteksi spike.
        """
        self.default_threshold = default_threshold

    # ----------------------------------------------------------------
    # Utilitas internal
    # ----------------------------------------------------------------

    @staticmethod
    def _format_date(date_val: Any) -> str:
        """Konversi nilai tanggal ke string YYYY-MM-DD."""
        if hasattr(date_val, "strftime"):
            return date_val.strftime("%Y-%m-%d")
        return str(date_val)

    @staticmethod
    def _detect_spikes_in_series(
        series: pd.Series,
        threshold_std: float,
    ) -> List[Dict[str, Any]]:
        """
        Deteksi indeks dan z-score untuk nilai yang melebihi threshold.

        Returns:
            List of (index_position, value, z_score)
        """
        if len(series) < 3:
            return []

        clean = series.fillna(0).astype(float)
        mean = clean.mean()
        std = clean.std()

        if std == 0:
            return []

        results: List[Dict[str, Any]] = []
        for idx, val in clean.items():
            z = (val - mean) / std
            if z >= threshold_std:
                results.append({"index": idx, "value": val, "z_score": z})

        return results

    # ----------------------------------------------------------------
    # Deteksi Spike Sentimen
    # ----------------------------------------------------------------

    def detect_sentiment_spikes(
        self,
        daily_df: pd.DataFrame,
        threshold_std: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identifikasi hari dengan lonjakan sentimen negatif/positif signifikan.

        Args:
            daily_df: DataFrame harian dengan kolom: date_only, total_post,
                      positive, negative.
            threshold_std: Threshold z-score (default: self.default_threshold).

        Returns:
            List dict: [{date, type, magnitude, count, pct, description}]
        """
        threshold = threshold_std or self.default_threshold
        spikes: List[Dict[str, Any]] = []

        if daily_df.empty or len(daily_df) < 3:
            logger.info("Data harian terlalu sedikit untuk deteksi spike sentimen.")
            return spikes

        sentiment_cols = [
            ("negative", "Negatif", "Lonjakan sentimen negatif terdeteksi"),
            ("positive", "Positif", "Lonjakan sentimen positif terdeteksi"),
        ]

        for col, label, desc_template in sentiment_cols:
            if col not in daily_df.columns:
                continue

            hits = self._detect_spikes_in_series(daily_df[col], threshold)
            for hit in hits:
                idx = hit["index"]
                row = daily_df.loc[idx]
                val = hit["value"]
                total = row.get("total_post", 1) or 1
                pct = round(val / total * 100, 1)
                date_str = self._format_date(row.get("date_only", idx))

                spikes.append({
                    "date":        date_str,
                    "type":        f"sentiment_{label.lower()}",
                    "magnitude":   round(float(hit["z_score"]), 2),
                    "count":       int(val),
                    "pct":         pct,
                    "description": f"{desc_template} pada {date_str}: "
                                   f"{int(val)} post ({pct}%), z-score={hit['z_score']:.2f}",
                })

        spikes.sort(key=lambda x: x["magnitude"], reverse=True)
        return spikes

    # ----------------------------------------------------------------
    # Deteksi Spike Emosi
    # ----------------------------------------------------------------

    def detect_emotion_spikes(
        self,
        emotion_daily_df: pd.DataFrame,
        threshold_std: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deteksi lonjakan emosi dari DataFrame tren emosi harian.

        Args:
            emotion_daily_df: DataFrame pivot harian dari
                              aggregate_emotion_daily() — kolom pertama = date_only,
                              sisa kolom = nama emosi.
            threshold_std: Threshold z-score.

        Returns:
            List dict: [{date, type, magnitude, count, pct, description}]
        """
        threshold = threshold_std or self.default_threshold
        spikes: List[Dict[str, Any]] = []

        if emotion_daily_df.empty or len(emotion_daily_df) < 3:
            logger.info("Data emosi harian terlalu sedikit untuk deteksi spike.")
            return spikes

        date_col = "date_only"
        if date_col not in emotion_daily_df.columns:
            # Ambil kolom pertama sebagai tanggal jika tidak ada date_only
            date_col = emotion_daily_df.columns[0]

        emotion_cols = [
            c for c in emotion_daily_df.columns if c != date_col
        ]

        for emotion in emotion_cols:
            hits = self._detect_spikes_in_series(
                emotion_daily_df[emotion], threshold,
            )
            for hit in hits:
                idx = hit["index"]
                row = emotion_daily_df.loc[idx]
                val = hit["value"]
                date_str = self._format_date(row.get(date_col, idx))

                # Hitung total post hari itu (jumlah semua emosi)
                total_day = sum(
                    row.get(c, 0) for c in emotion_cols
                )
                total_day = total_day or 1
                pct = round(val / total_day * 100, 1)

                spikes.append({
                    "date":        date_str,
                    "type":        f"emotion_{emotion.lower()}",
                    "magnitude":   round(float(hit["z_score"]), 2),
                    "count":       int(val),
                    "pct":         pct,
                    "description": f"Lonjakan emosi '{emotion}' pada {date_str}: "
                                   f"{int(val)} post ({pct}%), z-score={hit['z_score']:.2f}",
                })

        spikes.sort(key=lambda x: x["magnitude"], reverse=True)
        return spikes

    # ----------------------------------------------------------------
    # Deteksi Spike Engagement
    # ----------------------------------------------------------------

    def detect_engagement_spikes(
        self,
        daily_df: pd.DataFrame,
        threshold_std: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Deteksi anomali engagement harian.

        Args:
            daily_df: DataFrame harian dengan kolom: date_only, total_engagement.
            threshold_std: Threshold z-score.

        Returns:
            List dict: [{date, type, magnitude, count, pct, description}]
        """
        threshold = threshold_std or self.default_threshold
        spikes: List[Dict[str, Any]] = []

        if daily_df.empty or len(daily_df) < 3:
            logger.info("Data harian terlalu sedikit untuk deteksi spike engagement.")
            return spikes

        eng_col = "total_engagement"
        if eng_col not in daily_df.columns:
            logger.warning("Kolom '%s' tidak ditemukan.", eng_col)
            return spikes

        hits = self._detect_spikes_in_series(daily_df[eng_col], threshold)
        total_engagement_all = daily_df[eng_col].sum() or 1

        for hit in hits:
            idx = hit["index"]
            row = daily_df.loc[idx]
            val = hit["value"]
            date_str = self._format_date(row.get("date_only", idx))
            pct = round(val / total_engagement_all * 100, 1)

            spikes.append({
                "date":        date_str,
                "type":        "engagement",
                "magnitude":   round(float(hit["z_score"]), 2),
                "count":       int(val),
                "pct":         pct,
                "description": f"Lonjakan engagement pada {date_str}: "
                               f"{int(val):,} total engagement ({pct}% dari total), "
                               f"z-score={hit['z_score']:.2f}",
            })

        spikes.sort(key=lambda x: x["magnitude"], reverse=True)
        return spikes

    # ----------------------------------------------------------------
    # Deteksi Gabungan
    # ----------------------------------------------------------------

    def detect_all_spikes(
        self,
        daily_df: pd.DataFrame,
        emotion_daily_df: Optional[pd.DataFrame] = None,
        threshold_std: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Gabungan deteksi semua jenis spike: sentimen, emosi, engagement.

        Args:
            daily_df: DataFrame agregasi harian (dari aggregate_daily).
            emotion_daily_df: DataFrame tren emosi harian (opsional,
                              dari aggregate_emotion_daily).
            threshold_std: Threshold z-score.

        Returns:
            List dict gabungan, diurutkan berdasarkan magnitude tertinggi.
        """
        threshold = threshold_std or self.default_threshold
        all_spikes: List[Dict[str, Any]] = []

        # Sentimen
        all_spikes.extend(
            self.detect_sentiment_spikes(daily_df, threshold)
        )

        # Engagement
        all_spikes.extend(
            self.detect_engagement_spikes(daily_df, threshold)
        )

        # Emosi (opsional)
        if emotion_daily_df is not None and not emotion_daily_df.empty:
            all_spikes.extend(
                self.detect_emotion_spikes(emotion_daily_df, threshold)
            )

        # Urutkan semua berdasarkan magnitude
        all_spikes.sort(key=lambda x: x["magnitude"], reverse=True)

        logger.info("Total spike terdeteksi: %d", len(all_spikes))
        return all_spikes
