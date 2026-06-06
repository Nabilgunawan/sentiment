import math
from datetime import datetime


def generate_executive_summary(kpis, daily_df, account_df, top_keywords, df,
                                sentiment_spikes=None, viral_posts=None,
                                emotion_distribution=None, bigrams=None, trigrams=None):
    date_range = ""
    if "datetime" in df.columns and df["datetime"].notna().sum() > 0:
        valid_dates = df["datetime"].dropna()
        start_date = valid_dates.min()
        end_date = valid_dates.max()
        if hasattr(start_date, "strftime"):
            date_range = f"{start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')}"
        else:
            date_range = f"{start_date} - {end_date}"

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

    # ── Opening Section ──
    parts = []
    parts.append(f"# Laporan Analisis Sentimen Media Sosial")
    parts.append(f"")
    parts.append(f"**Periode:** {date_range} | **Total Post:** {total:,} | **Akun Unik:** {unique_acc:,}")
    parts.append(f"")

    # Opening narrative
    if total > 0 and unique_acc > 0:
        avg_post_per_account = round(total / unique_acc, 1)
        parts.append(f"## 📊 Gambaran Umum")
        if pos_pct > 50:
            opening = (
                f"Selama periode **{date_range}**, terdapat **{total:,} postingan** "
                f"dari **{unique_acc:,} akun** yang berhasil dihimpun. "
                f"Mayoritas percakapan menunjukkan sentimen **positif ({pos_pct}%)**, "
                f"mengindikasikan bahwa publik memberikan respons yang baik terhadap topik yang beredar. "
                f"Rata-rata setiap akun memproduksi **{avg_post_per_account} postingan**, "
                f"dengan total engagement mencapai **{total_eng:,} interaksi**."
            )
        elif neg_pct > 50:
            opening = (
                f"Sepanjang periode **{date_range}**, tercatat **{total:,} postingan** "
                f"dari **{unique_acc:,} akun** membahas topik ini. "
                f"Dominasi sentimen **negatif ({neg_pct}%)** menandakan adanya gelombang "
                f"kekhawatiran atau kekecewaan publik yang cukup signifikan. "
                f" dengan total **{total_eng:,} interaksi**, situasi ini memerlukan perhatian serius."
            )
        else:
            opening = (
                f"Dalam rentang **{date_range}**, teridentifikasi **{total:,} postingan** "
                f"dari **{unique_acc:,} akun** yang membahas topik terkait. "
                f"Distribusi sentimen relatif berimbang dengan **{pos_pct}% positif**, "
                f"**{neu_pct}% netral**, dan **{neg_pct}% negatif**. "
                f"Total engagement tercatat sebesar **{total_eng:,} interaksi**, "
                f"menunjukkan tingkat diskusi yang aktif di media sosial."
            )
        parts.append(opening)
        parts.append(f"")

    # ── Sentiment Findings ──
    parts.append(f"## 📈 Temuan Sentimen")

    if pos_pct > neg_pct * 2 and pos_pct > 40:
        parts.append(
            f"Sentimen publik didominasi oleh pandangan **positif ({pos_pct}%)**, "
            f"jauh melampaui sentimen negatif ({neg_pct}%). "
            f"Hal ini mengindikasikan bahwa **{positive:,} dari {total:,} postingan** "
            f"menunjukkan dukungan, apresiasi, atau kepuasan. "
            f"Hanya {negative:,} postingan ({neg_pct}%) yang bersifat negatif."
        )
    elif neg_pct > pos_pct * 2 and neg_pct > 40:
        parts.append(
            f"Terdapat gelombang sentimen **negatif ({neg_pct}%)** yang mendominasi, "
            f"dengan **{negative:,} dari {total:,} postingan** berisi keluhan, kritik, "
            f"atau kekecewaan. Sentimen positif hanya tercatat **{pos_pct}%** "
            f"({positive:,} postingan). Hal ini menjadi sinyal awal yang perlu direspons."
        )
    elif abs(pos_pct - neg_pct) < 10:
        parts.append(
            f"Sentimen publik terbagi hampir merata antara positif ({pos_pct}%) "
            f"dan negatif ({neg_pct}%), dengan sisanya bersikap netral ({neu_pct}%). "
            f"Polarisasi opini ini menunjukkan bahwa topik bersifat kontroversial "
            f"atau memicu perdebatan di kalangan publik."
        )
    else:
        parts.append(
            f"Distribusi sentimen menunjukkan **{pos_pct}% positif**, "
            f"**{neu_pct}% netral**, dan **{neg_pct}% negatif**. "
            f"Mayoritas pengguna ({pos_pct + neu_pct}%) memberikan respons yang "
            f"cenderung tidak negatif, meskipun terdapat {negative:,} postingan "
            f"yang perlu mendapat perhatian."
        )

    avg_engagement = total_eng / total if total > 0 else 0
    if avg_engagement > 100:
        parts.append(
            f"Tingkat engagement sangat tinggi dengan rata-rata **{avg_engagement:.1f}** "
            f"interaksi per postingan, menandakan topik ini sangat relevan dan "
            f"menggerakkan partisipasi aktif publik."
        )
    elif avg_engagement > 20:
        parts.append(
            f"Rata-rata engagement mencapai **{avg_engagement:.1f}** interaksi per post, "
            f"menunjukkan diskusi yang cukup hidup di platform."
        )
    else:
        parts.append(
            f"Engagement rata-rata sebesar **{avg_engagement:.1f}** per post, "
            f"dapat ditingkatkan dengan strategi konten yang lebih partisipatif."
        )
    parts.append(f"")

    # ── Spike Analysis ──
    parts.append(f"## ⚡ Spike Analysis")
    if sentiment_spikes and len(sentiment_spikes) > 0:
        for spike in sentiment_spikes[:3]:
            direction = "meningkat tajam" if spike["type"] == "Negatif" else "melonjak signifikan"
            emoji = "🔴" if spike["type"] == "Negatif" else "🟢"
            parts.append(
                f"{emoji} **{spike['date']}** — Sentimen **{spike['type']}** {direction} "
                f"(magnitude: {spike['magnitude']}σ, {spike['count']} post, {spike['pct']}% dari total harian)."
            )
    else:
        parts.append(
            f"Tidak terdeteksi lonjakan sentimen yang signifikan (di atas 2 standar deviasi) "
            f"selama periode analisis. Sentimen bergerak relatif stabil dari hari ke hari."
        )
    parts.append(f"")

    # ── Dominant Emotion ──
    parts.append(f"## 🎭 Emosi Dominan")
    if emotion_distribution:
        sorted_emotions = sorted(
            emotion_distribution.items(),
            key=lambda x: x[1]["percentage"],
            reverse=True
        )
        top_emotion = sorted_emotions[0] if sorted_emotions else None
        if top_emotion and top_emotion[1]["percentage"] > 0:
            parts.append(
                f"Emosi yang paling dominan dirasakan publik adalah **{top_emotion[0]}** "
                f"({top_emotion[1]['percentage']}% dari total post). "
            )
            emotion_desc = {
                "Marah": "menunjukkan tingkat frustrasi dan kemarahan publik yang perlu diredam",
                "Takut": "mengindikasikan kekhawatiran dan kecemasan yang meluas",
                "Jijik": "menunjukkan penolakan moral atau estetika terhadap isu terkait",
                "Sedih": "menggambarkan kekecewaan dan kesedihan mendalam publik",
                "Antisipasi": "publik menantikan dengan penuh harapan perkembangan ke depan",
                "Percaya": "publik menunjukkan kepercayaan dan optimisme yang tinggi",
                "Terkejut": "publik dikejutkan oleh informasi atau peristiwa yang tidak terduga",
                "Senang": "publik merespons dengan kegembiraan dan kepuasan",
            }
            desc = emotion_desc.get(top_emotion[0], "")
            if desc:
                parts.append(f"Hal ini {desc}.")

            # List all emotions
            emotion_line = []
            for em, data in sorted_emotions:
                if data["percentage"] > 0:
                    emotion_line.append(f"**{em}**: {data['percentage']}%")
            if emotion_line:
                parts.append(f"Distribusi emosi lengkap: {', '.join(emotion_line)}.")
        else:
            parts.append("Distribusi emosi belum tersedia untuk dataset ini.")
    else:
        parts.append("Distribusi emosi belum tersedia untuk dataset ini.")
    parts.append(f"")

    # ── Top Influencers ──
    parts.append(f"## 👤 Akun Paling Berpengaruh")
    if account_df is not None and len(account_df) > 0:
        top_accounts = account_df.head(5)
        parts.append("Berikut adalah akun dengan aktivitas dan engagement tertinggi:")
        for idx, (_, row) in enumerate(top_accounts.iterrows(), 1):
            sent_emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
            emoji = sent_emoji.get(row.get("dominant_sentiment", "Neutral"), "⚪")
            parts.append(
                f"{idx}. {emoji} **{row['X akun']}** — {int(row['total_post'])} post, "
                f"{int(row['total_engagement']):,} engagement, "
                f"sentimen dominan: {row.get('dominant_sentiment', 'N/A')}"
            )
        parts.append(f"Total {len(account_df)} akun terdeteksi selama periode analisis.")
    parts.append(f"")

    # ── Viral Content ──
    parts.append(f"## 🔥 Konten Paling Viral")
    if viral_posts and len(viral_posts) > 0:
        parts.append(
            f"Berikut adalah **{len(viral_posts)} postingan** dengan engagement tertinggi "
            f"yang menjadi pusat perhatian publik:"
        )
        for idx, post in enumerate(viral_posts[:5], 1):
            sent_emoji = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
            emoji = sent_emoji.get(post.get("sentiment", "Neutral"), "⚪")
            konten = post.get("konten", "")
            konten_short = konten[:80] + "..." if len(konten) > 80 else konten
            parts.append(
                f"{idx}. {emoji} **{post['akun']}** — \"{konten_short}\"\n"
                f"   👍 {post['likes']:,} | 🔄 {post['repost']:,} | 💬 {post['komentar']:,} | 👁 {post['views']:,} | "
                f"Engagement Score: {post['engagement_score']:.1f}"
            )
    else:
        parts.append("Data viral post belum tersedia untuk dataset ini.")
    parts.append(f"")

    # ── Keyword Analysis ──
    if bigrams or trigrams:
        parts.append(f"## 🔍 Analisis Frasa")
        if bigrams:
            top_bi = bigrams[:5]
            parts.append(
                f"**Bigram terpopuler:** " +
                ", ".join(f"\"{w}\" ({c}x)" for w, c in top_bi)
            )
        if trigrams:
            top_tri = trigrams[:5]
            parts.append(
                f"**Trigram terpopuler:** " +
                ", ".join(f"\"{w}\" ({c}x)" for w, c in top_tri)
            )
        parts.append(f"")

    if top_keywords:
        parts.append(f"## 🏷️ Top Keywords")
        keywords_str = ", ".join(f"**{w}** ({c})" for w, c in top_keywords[:10])
        parts.append(f"Kata kunci yang paling sering muncul dalam diskusi publik: {keywords_str}.")
        parts.append(f"")

    # ── Recommendations ──
    parts.append(f"## 💡 Rekomendasi")

    recommendations = []
    if neg_pct > 30:
        recommendations.append(
            f"🔴 **Tingkat sentimen negatif tinggi ({neg_pct}%).** "
            f"Segera lakukan investigasi mendalam terhadap akar penyebab ketidakpuasan publik. "
            f"Prioritaskan respons pada {negative:,} postingan negatif yang teridentifikasi."
        )
    if pos_pct > 50:
        recommendations.append(
            f"🟢 **Sentimen positif mendominasi ({pos_pct}%).** "
            f"Pertahankan strategi komunikasi yang sudah berjalan. "
            f"Amplifikasi konten positif melalui kanal resmi untuk memperkuat persepsi baik."
        )
    if emotion_distribution:
        sorted_em = sorted(emotion_distribution.items(), key=lambda x: x[1]["percentage"], reverse=True)
        if sorted_em and sorted_em[0][1]["percentage"] > 0:
            top_em_name = sorted_em[0][0]
            em_recs = {
                "Marah": "Redam kemarahan dengan komunikasi yang transparan, empati, dan solusi konkret.",
                "Takut": "Berikan informasi yang menenangkan dan edukatif untuk meredakan kecemasan publik.",
                "Jijik": "Lakukan perbaikan fundamental untuk menghilangkan sumber rasa jijik publik.",
                "Sedih": "Tunjukkan empati dan komitmen untuk memperbaiki situasi yang mengecewakan.",
                "Antisipasi": "Penuhi ekspektasi publik dengan memberikan update dan inovasi secara konsisten.",
                "Percaya": "Jaga kepercayaan publik dengan transparansi dan konsistensi.",
                "Terkejut": "Manfaatkan momen perhatian publik untuk menyampaikan pesan kunci.",
                "Senang": "Manfaatkan momentum positif untuk memperkuat hubungan dengan audiens.",
            }
            if top_em_name in em_recs:
                recommendations.append(f"🎯 **Emosi dominan: {top_em_name}.** {em_recs[top_em_name]}")

    if total_views > 10000:
        recommendations.append(
            f"👁 **Visibilitas tinggi ({total_views:,} views).** "
            f"Optimalkan momen ini dengan meluncurkan kampanye atau konten strategis."
        )
    elif total_views < 1000 and total > 10:
        recommendations.append(
            f"📢 **Visibilitas masih terbatas.** "
            f"Tingkatkan distribusi konten dan pertimbangkan strategi paid promotion."
        )

    if not recommendations:
        recommendations.append(
            "Pantau terus perkembangan percakapan dan lakukan analisis berkala "
            "untuk mengidentifikasi perubahan sentimen sejak dini."
        )

    for rec in recommendations:
        parts.append(f"- {rec}")
    parts.append(f"")

    # Confidence note
    if kpis.get("avg_confidence", 1.0) < 0.5:
        parts.append(f"---")
        parts.append(
            f"> ⚠️ **Catatan:** Confidence score rata-rata **{kpis['avg_confidence']:.2f}**. "
            f"Hasil analisis perlu diinterpretasikan dengan hati-hati. "
            f"Disarankan menggunakan API key DeepSeek untuk hasil yang lebih akurat."
        )

    return "\n".join(parts)
