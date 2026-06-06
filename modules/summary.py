from datetime import datetime


def generate_executive_summary(kpis, daily_df, account_df, top_keywords, df):
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

    if pos_pct > neg_pct * 2 and pos_pct > 50:
        overall = "positif"
        tone = "baik"
        recommendation = "Pertahankan strategi konten yang sudah berjalan."
    elif neg_pct > pos_pct * 2 and neg_pct > 50:
        overall = "negatif"
        tone = "kurang baik"
        recommendation = "Perlu evaluasi strategi komunikasi dan identifikasi penyebab sentimen negatif."
    elif pos_pct > neg_pct:
        overall = "cenderung positif"
        tone = "cukup baik"
        recommendation = "Terus tingkatkan engagement dengan mempertahankan kualitas konten."
    elif neg_pct > pos_pct:
        overall = "cenderung negatif"
        tone = "perlu perhatian"
        recommendation = "Perhatikan keluhan publik dan lakukan perbaikan responsif."
    else:
        overall = "berimbang"
        tone = "netral"
        recommendation = "Identifikasi peluang untuk membedakan konten dari kompetitor."

    top_accounts = account_df.head(5)
    top_accounts_str = "; ".join(
        f"{row['X akun']} ({int(row['total_post'])} post, {int(row['total_engagement'])} engagement)"
        for _, row in top_accounts.iterrows()
    )

    busiest_day = ""
    if len(daily_df) > 0:
        max_row = daily_df.loc[daily_df["total_post"].idxmax()]
        busiest_day = f"{max_row['date_only']} ({int(max_row['total_post'])} post)"

    keywords_str = ", ".join(w for w, _ in top_keywords[:10])

    summary_parts = [
        f"# Executive Summary",
        f"",
        f"## Ringkasan Umum",
        f"Periode analisis: {date_range}" if date_range else "",
        f"Total {total} post dari {kpis['unique_accounts']} akun dianalisis.",
        f"Engagement total: {kpis['total_engagement']:,} interaksi.",
        f"",
        f"## Sentimen",
        f"Distribusi sentimen: Positif {pos_pct}% ({positive} post), Netral {neu_pct}% ({neutral} post), Negatif {neg_pct}% ({negative} post).",
        f"Secara keseluruhan, sentimen pengguna {overall} ({tone}).",
        f"",
        f"## Insight",
    ]

    if pos_pct > 50:
        summary_parts.append(f"- Mayoritas post ({pos_pct}%) mendapat respons positif dari audiens.")
    if neg_pct > 30:
        summary_parts.append(f"- Terdapat {neg_pct}% post dengan sentimen negatif yang perlu ditindaklanjuti.")
    if neutral > 30:
        summary_parts.append(f"- Sebanyak {neu_pct}% post bersifat netral, berpotensi untuk dikonversi menjadi engagement positif.")

    avg_engagement = kpis["total_engagement"] / total if total > 0 else 0
    if avg_engagement > 100:
        summary_parts.append(f"- Rata-rata engagement per post cukup tinggi ({avg_engagement:.1f}), menunjukkan interaksi aktif.")
    elif avg_engagement > 10:
        summary_parts.append(f"- Rata-rata engagement per post: {avg_engagement:.1f}.")
    else:
        summary_parts.append(f"- Rata-rata engagement per post masih rendah ({avg_engagement:.1f}), perlu strategi boosting.")

    if top_accounts_str:
        summary_parts.append(f"")
        summary_parts.append(f"## Akun Paling Aktif")
        summary_parts.append(f"Akun dengan aktivitas tertinggi: {top_accounts_str}.")

    if busiest_day:
        summary_parts.append(f"")
        summary_parts.append(f"Hari dengan post terbanyak: {busiest_day}.")

    if keywords_str:
        summary_parts.append(f"")
        summary_parts.append(f"## Top Keywords")
        summary_parts.append(f"Kata kunci yang sering muncul: {keywords_str}.")

    summary_parts.append(f"")
    summary_parts.append(f"## Rekomendasi")
    summary_parts.append(f"{recommendation}")

    if kpis["avg_confidence"] < 0.5:
        summary_parts.append(f"")
        summary_parts.append(f"Catatan: Confidence score rata-rata {kpis['avg_confidence']:.2f}. "
                              f"Hasil analisis perlu diinterpretasikan dengan hati-hati.")

    return "\n".join(summary_parts)
