"""
services/summary_service.py
-----------------------------
Service untuk generate executive summary.

Strategi:
1. Jika DeepSeek API tersedia → gunakan AI untuk narrative summary
2. Jika tidak → gunakan template-based summary (rule-based)

Output: Markdown string berisi laporan eksekutif lengkap.
"""
import math
import logging
from datetime import datetime
from typing import Optional

from .deepseek_client import is_api_available, generate_ai_summary

logger = logging.getLogger(__name__)


def generate_executive_summary(
    kpis: dict,
    daily_df,
    account_df,
    top_keywords: list,
    df,
    sentiment_spikes: Optional[list] = None,
    viral_posts: Optional[list] = None,
    emotion_distribution: Optional[dict] = None,
    sarcasm_distribution: Optional[dict] = None,
    topics: Optional[list] = None,
    bigrams: Optional[list] = None,
    trigrams: Optional[list] = None,
    weighted_sentiment: Optional[dict] = None,
    sentiment_index: float = 0.0,
    use_ai: bool = True,
) -> str:
    """
    Generate executive summary — AI-powered atau template-based.

    Args:
        kpis: KPI metrics dict
        daily_df: Daily aggregation DataFrame
        account_df: Account aggregation DataFrame
        top_keywords: List of (word, count) tuples
        df: Full analysis DataFrame
        sentiment_spikes: List of spike dicts
        viral_posts: List of viral post dicts
        emotion_distribution: Emotion distribution dict
        sarcasm_distribution: Sarcasm distribution dict
        topics: List of topic dicts
        bigrams: List of (phrase, count) tuples
        trigrams: List of (phrase, count) tuples
        weighted_sentiment: Weighted sentiment dict
        sentiment_index: Sentiment index float
        use_ai: Whether to try AI summary first

    Returns:
        Markdown string executive summary
    """
    import json
    import pandas as pd

    # Determine date range
    date_range = ""
    if "datetime" in df.columns and df["datetime"].notna().sum() > 0:
        valid_dates = df["datetime"].dropna()
        start_date = valid_dates.min()
        end_date = valid_dates.max()
        if hasattr(start_date, "strftime"):
            date_range = f"{start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}"
        else:
            date_range = f"{start_date} - {end_date}"

    # Try AI summary first
    if use_ai and is_api_available():
        try:
            analysis_data = {
                "kpis": kpis,
                "sentiment_index": sentiment_index,
                "emotion_distribution": emotion_distribution or {},
                "sarcasm_distribution": sarcasm_distribution or {},
                "topics": topics or [],
                "sentiment_spikes": sentiment_spikes or [],
                "viral_posts": viral_posts or [],
                "accounts": json.loads(account_df.head(10).to_json(orient="records"))
                    if account_df is not None and len(account_df) > 0 else [],
                "keywords": [{"word": w, "count": c} for w, c in top_keywords] if top_keywords else [],
                "bigrams": [{"phrase": w, "count": c} for w, c in bigrams] if bigrams else [],
                "trigrams": [{"phrase": w, "count": c} for w, c in trigrams] if trigrams else [],
                "date_range": date_range,
                "total_rows": len(df),
                "analysis_mode": "IndoBERT Transformer",
                "weighted_sentiment": weighted_sentiment or {},
            }
            ai_summary = generate_ai_summary(analysis_data)
            if ai_summary:
                logger.info("Executive summary berhasil di-generate oleh DeepSeek AI")
                return ai_summary
        except Exception as e:
            logger.warning(f"AI summary gagal, fallback ke template: {e}")

    # Fallback: Template-based summary
    logger.info("Menggunakan template-based executive summary")
    return _generate_template_summary(
        kpis, daily_df, account_df, top_keywords, df,
        sentiment_spikes, viral_posts, emotion_distribution,
        sarcasm_distribution, topics, bigrams, trigrams,
        weighted_sentiment, sentiment_index, date_range,
    )


def _generate_template_summary(
    kpis, daily_df, account_df, top_keywords, df,
    sentiment_spikes, viral_posts, emotion_distribution,
    sarcasm_distribution, topics, bigrams, trigrams,
    weighted_sentiment, sentiment_index, date_range,
) -> str:
    """Generate template-based executive summary (fallback)."""
    total = kpis["total_post"]
    positive = kpis["positive"]
    neutral = kpis["neutral"]
    negative = kpis["negative"]
    pos_pct = kpis["positive_pct"]
    neu_pct = kpis["neutral_pct"]
    neg_pct = kpis["negative_pct"]
    unique_acc = kpis["unique_accounts"]
    total_eng = kpis["total_engagement"]
    total_views = kpis["total_views"]
    sarcasm_count = kpis.get("sarcasm_count", 0)
    sarcasm_pct = kpis.get("sarcasm_pct", 0)

    parts = []
    parts.append("# Laporan Analisis Sentimen Media Sosial")
    parts.append("")
    parts.append(
        f"**Periode:** {date_range} | **Total Post:** {total:,} | "
        f"**Akun Unik:** {unique_acc:,} | **Metode:** IndoBERT Transformer"
    )
    parts.append("")

    # ── Gambaran Umum ──
    if total > 0 and unique_acc > 0:
        avg_post = round(total / unique_acc, 1)
        parts.append("## 📊 Gambaran Umum")
        if pos_pct > 50:
            parts.append(
                f"Selama periode **{date_range}**, terdapat **{total:,} postingan** "
                f"dari **{unique_acc:,} akun** yang berhasil dianalisis menggunakan model IndoBERT. "
                f"Mayoritas percakapan menunjukkan sentimen **positif ({pos_pct}%)**, "
                f"mengindikasikan respons publik yang baik. "
                f"Rata-rata setiap akun memproduksi **{avg_post} postingan** "
                f"dengan total engagement **{total_eng:,} interaksi**."
            )
        elif neg_pct > 50:
            parts.append(
                f"Sepanjang periode **{date_range}**, tercatat **{total:,} postingan** "
                f"dari **{unique_acc:,} akun**. Dominasi sentimen **negatif ({neg_pct}%)** "
                f"menandakan gelombang kekhawatiran publik yang signifikan. "
                f"Total **{total_eng:,} interaksi** menunjukkan intensitas diskusi yang tinggi."
            )
        else:
            parts.append(
                f"Dalam rentang **{date_range}**, teridentifikasi **{total:,} postingan** "
                f"dari **{unique_acc:,} akun**. Distribusi sentimen: "
                f"**{pos_pct}% positif**, **{neu_pct}% netral**, **{neg_pct}% negatif**. "
                f"Total engagement: **{total_eng:,} interaksi**."
            )
        parts.append("")

    # ── Sentiment Index ──
    parts.append("## 📈 Temuan Sentimen")
    parts.append(f"**Sentiment Index: {sentiment_index}** (skala -100 s.d. +100)")
    parts.append("")

    if pos_pct > neg_pct * 2 and pos_pct > 40:
        parts.append(
            f"Sentimen publik didominasi **positif ({pos_pct}%)**, jauh melampaui negatif ({neg_pct}%). "
            f"**{positive:,} dari {total:,}** postingan menunjukkan dukungan dan apresiasi."
        )
    elif neg_pct > pos_pct * 2 and neg_pct > 40:
        parts.append(
            f"Gelombang sentimen **negatif ({neg_pct}%)** mendominasi dengan "
            f"**{negative:,} postingan** berisi keluhan atau kritik."
        )
    elif abs(pos_pct - neg_pct) < 10:
        parts.append(
            f"Sentimen terbagi merata: positif {pos_pct}%, negatif {neg_pct}%, netral {neu_pct}%. "
            f"Polarisasi ini menunjukkan topik yang kontroversial."
        )
    else:
        parts.append(
            f"Distribusi: **{pos_pct}% positif**, **{neu_pct}% netral**, **{neg_pct}% negatif**."
        )
    parts.append("")

    # ── Emosi ──
    parts.append("## 🎭 Emosi Dominan")
    if emotion_distribution:
        sorted_emotions = sorted(
            emotion_distribution.items(),
            key=lambda x: x[1].get("percentage", 0) if isinstance(x[1], dict) else 0,
            reverse=True,
        )
        top_emotion = sorted_emotions[0] if sorted_emotions else None
        if top_emotion:
            em_name = top_emotion[0]
            em_data = top_emotion[1] if isinstance(top_emotion[1], dict) else {"percentage": 0}
            parts.append(
                f"Emosi dominan: **{em_name}** ({em_data.get('percentage', 0)}%). "
            )
            emotion_desc = {
                "Marah": "menunjukkan frustrasi dan kemarahan publik",
                "Takut": "mengindikasikan kekhawatiran yang meluas",
                "Jijik": "menunjukkan penolakan terhadap isu terkait",
                "Sedih": "menggambarkan kekecewaan mendalam",
                "Terkejut": "publik dikejutkan oleh informasi yang beredar",
                "Senang": "publik merespons dengan kegembiraan",
                "Netral": "tidak ada emosi yang sangat dominan",
            }
            if em_name in emotion_desc:
                parts.append(f"Hal ini {emotion_desc[em_name]}.")

            em_lines = [
                f"**{em}**: {d.get('percentage', 0) if isinstance(d, dict) else 0}%"
                for em, d in sorted_emotions if (isinstance(d, dict) and d.get("percentage", 0) > 0)
            ]
            if em_lines:
                parts.append(f"Distribusi lengkap: {', '.join(em_lines)}.")
    else:
        parts.append("Data emosi belum tersedia.")
    parts.append("")

    # ── Sarkasme ──
    parts.append("## 🎪 Deteksi Sarkasme")
    if sarcasm_count > 0:
        parts.append(
            f"Terdeteksi **{sarcasm_count:,} postingan sarkastik** ({sarcasm_pct}% dari total). "
            f"Konten sarkastik ini perlu perhatian khusus karena sentimen permukaan "
            f"seringkali berlawanan dengan makna sebenarnya."
        )
    else:
        parts.append("Tidak terdeteksi konten sarkastik yang signifikan.")
    parts.append("")

    # ── Topik ──
    parts.append("## 📌 Topik Utama")
    if topics:
        for i, t in enumerate(topics[:5], 1):
            parts.append(f"{i}. **{t.get('name', 'N/A')}** — {t.get('frequency', 0)} post")
    else:
        parts.append("Ekstraksi topik belum tersedia.")
    parts.append("")

    # ── Spike ──
    parts.append("## ⚡ Spike Analysis")
    if sentiment_spikes and len(sentiment_spikes) > 0:
        for spike in sentiment_spikes[:3]:
            direction = "meningkat tajam" if spike["type"] == "Negatif" else "melonjak"
            emoji = "🔴" if spike["type"] == "Negatif" else "🟢"
            parts.append(
                f"{emoji} **{spike['date']}** — Sentimen **{spike['type']}** {direction} "
                f"(magnitude: {spike['magnitude']}σ, {spike['count']} post, {spike['pct']}%)"
            )
    else:
        parts.append("Tidak ada lonjakan sentimen signifikan selama periode analisis.")
    parts.append("")

    # ── Influencer ──
    parts.append("## 👤 Akun Berpengaruh")
    if account_df is not None and len(account_df) > 0:
        for idx, (_, row) in enumerate(account_df.head(5).iterrows(), 1):
            sent_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
            emoji = sent_map.get(row.get("dominant_sentiment", ""), "⚪")
            parts.append(
                f"{idx}. {emoji} **{row['X akun']}** — {int(row['total_post'])} post, "
                f"{int(row['total_engagement']):,} engagement"
            )
    parts.append("")

    # ── Viral ──
    parts.append("## 🔥 Konten Viral")
    if viral_posts:
        for idx, post in enumerate(viral_posts[:5], 1):
            sent_map = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
            emoji = sent_map.get(post.get("sentiment", ""), "⚪")
            konten = post.get("konten", "")[:80]
            if len(post.get("konten", "")) > 80:
                konten += "..."
            parts.append(
                f"{idx}. {emoji} **{post['akun']}** — \"{konten}\"\n"
                f"   👍 {post.get('likes', 0):,} | 🔄 {post.get('repost', 0):,} | "
                f"💬 {post.get('komentar', 0):,} | Score: {post.get('engagement_score', 0):.1f}"
            )
    parts.append("")

    # ── Frasa ──
    if bigrams or trigrams:
        parts.append("## 🔍 Analisis Frasa")
        if bigrams:
            top_bi = bigrams[:5]
            parts.append("**Bigram:** " + ", ".join(f'"{w}" ({c}x)' for w, c in top_bi))
        if trigrams:
            top_tri = trigrams[:5]
            parts.append("**Trigram:** " + ", ".join(f'"{w}" ({c}x)' for w, c in top_tri))
        parts.append("")

    # ── Keywords ──
    if top_keywords:
        parts.append("## 🏷️ Keywords Utama")
        kw_str = ", ".join(f"**{w}** ({c})" for w, c in top_keywords[:10])
        parts.append(f"Kata kunci terpopuler: {kw_str}")
        parts.append("")

    # ── Rekomendasi ──
    parts.append("## 💡 Rekomendasi")
    recs = []
    if neg_pct > 30:
        recs.append(
            f"🔴 **Sentimen negatif tinggi ({neg_pct}%).** "
            f"Investigasi akar penyebab ketidakpuasan pada {negative:,} postingan negatif."
        )
    if pos_pct > 50:
        recs.append(
            f"🟢 **Sentimen positif dominan ({pos_pct}%).** "
            f"Pertahankan dan amplifikasi strategi komunikasi yang berjalan."
        )
    if sarcasm_pct > 10:
        recs.append(
            f"🎪 **Tingkat sarkasme tinggi ({sarcasm_pct}%).** "
            f"Perhatikan konten sarkastik yang mungkin menyembunyikan sentimen negatif."
        )
    if not recs:
        recs.append("Pantau perkembangan percakapan secara berkala.")

    for rec in recs:
        parts.append(f"- {rec}")
    parts.append("")

    # Confidence
    avg_conf = kpis.get("avg_confidence", 1.0)
    if avg_conf < 0.6:
        parts.append("---")
        parts.append(
            f"> ⚠️ **Catatan:** Confidence rata-rata **{avg_conf:.2f}**. "
            f"Disarankan melakukan fine-tuning model untuk akurasi lebih tinggi."
        )

    return "\n".join(parts)
