"""
chart_renderer.py
-----------------
Modul rendering chart server-side menggunakan matplotlib untuk embedding
di laporan DOCX/XLSX. Semua fungsi mengembalikan bytes PNG via io.BytesIO.

Tema visual: dark background (#0f172a), font Inter/sans-serif, 150 DPI.
"""
from __future__ import annotations

import io
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – harus sebelum import pyplot

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)

# ── Konstanta Warna ───────────────────────────────────────────────────────────
COLOR_POSITIVE = "#22c55e"
COLOR_NEUTRAL = "#64748b"
COLOR_NEGATIVE = "#ef4444"

EMOTION_COLORS: dict[str, str] = {
    "Marah": "#ef4444",
    "Senang": "#22c55e",
    "Sedih": "#3b82f6",
    "Takut": "#a855f7",
    "Jijik": "#f97316",
    "Terkejut": "#eab308",
    "Netral": "#64748b",
}

DARK_BG = "#0f172a"
DARK_FG = "#e2e8f0"
GRID_COLOR = "#334155"

# ── Konfigurasi Default ──────────────────────────────────────────────────────
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 150
FONT_FAMILY = "Inter, Segoe UI, sans-serif"


# ── Helper Internal ───────────────────────────────────────────────────────────
def _apply_dark_theme() -> None:
    """Terapkan tema gelap kustom ke matplotlib rc."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": DARK_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": DARK_FG,
        "text.color": DARK_FG,
        "xtick.color": DARK_FG,
        "ytick.color": DARK_FG,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial"],
        "legend.facecolor": DARK_BG,
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": DARK_FG,
    })


def _fig_to_bytes(fig: plt.Figure, dpi: int = DEFAULT_DPI) -> bytes:
    """Konversi Figure matplotlib ke bytes PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _safe_pct(value: float | int | None, default: float = 0.0) -> float:
    """Validasi persentase, kembalikan default jika None/NaN."""
    if value is None:
        return default
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


# ── Public API ────────────────────────────────────────────────────────────────

def render_sentiment_donut(
    pos_pct: float,
    neu_pct: float,
    neg_pct: float,
) -> bytes:
    """
    Render donut chart distribusi sentimen.

    Parameters
    ----------
    pos_pct : float
        Persentase sentimen positif (0-100).
    neu_pct : float
        Persentase sentimen netral (0-100).
    neg_pct : float
        Persentase sentimen negatif (0-100).

    Returns
    -------
    bytes
        PNG image bytes.
    """
    _apply_dark_theme()

    pos_pct = _safe_pct(pos_pct)
    neu_pct = _safe_pct(neu_pct)
    neg_pct = _safe_pct(neg_pct)

    # Hindari chart kosong
    total = pos_pct + neu_pct + neg_pct
    if total == 0:
        pos_pct = neu_pct = neg_pct = 33.33

    sizes = [pos_pct, neu_pct, neg_pct]
    colors = [COLOR_POSITIVE, COLOR_NEUTRAL, COLOR_NEGATIVE]
    labels = ["Positif", "Netral", "Negatif"]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"width": 0.4, "edgecolor": DARK_BG, "linewidth": 2},
        textprops={"fontsize": 13, "color": DARK_FG},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.set_title("Distribusi Sentimen", fontsize=16, fontweight="bold",
                 pad=20, color=DARK_FG)

    logger.debug("Sentiment donut chart rendered successfully.")
    return _fig_to_bytes(fig)


def render_emotion_bar(emotion_distribution: dict[str, int | float]) -> bytes:
    """
    Render horizontal bar chart distribusi emosi.

    Parameters
    ----------
    emotion_distribution : dict
        Mapping nama emosi → jumlah/persentase.
        Contoh: {"Marah": 120, "Senang": 350, ...}

    Returns
    -------
    bytes
        PNG image bytes.
    """
    _apply_dark_theme()

    if not emotion_distribution:
        emotion_distribution = {e: 0 for e in EMOTION_COLORS}

    emotions = list(emotion_distribution.keys())
    values = [float(emotion_distribution.get(e, 0)) for e in emotions]
    colors = [EMOTION_COLORS.get(e, "#94a3b8") for e in emotions]

    # Urutkan dari terbesar ke terkecil (atas = terbesar)
    sorted_indices = np.argsort(values)
    emotions = [emotions[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax.barh(emotions, values, color=colors, height=0.6, edgecolor="none")

    # Tambah label nilai di ujung bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=11, color=DARK_FG)

    ax.set_xlabel("Jumlah", fontsize=12)
    ax.set_title("Distribusi Emosi", fontsize=16, fontweight="bold",
                 pad=15, color=DARK_FG)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    logger.debug("Emotion bar chart rendered successfully.")
    return _fig_to_bytes(fig)


def render_sentiment_timeline(daily_df: Any) -> bytes:
    """
    Render line chart tren sentimen harian.

    Parameters
    ----------
    daily_df : pd.DataFrame | list[dict]
        Harus memiliki kolom/key: date, positive, neutral, negative.
        Jika list[dict], akan dikonversi ke DataFrame.

    Returns
    -------
    bytes
        PNG image bytes.
    """
    import pandas as pd

    _apply_dark_theme()

    if daily_df is None:
        logger.warning("daily_df is None, returning empty chart.")
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.text(0.5, 0.5, "Data tidak tersedia", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=DARK_FG)
        ax.set_title("Tren Sentimen Harian", fontsize=16, fontweight="bold",
                     color=DARK_FG)
        return _fig_to_bytes(fig)

    if isinstance(daily_df, list):
        daily_df = pd.DataFrame(daily_df)

    df = daily_df.copy()

    # Pastikan kolom date ada dan terurut
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        x = df["date"]
    else:
        x = range(len(df))

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    for col, color, label in [
        ("positive", COLOR_POSITIVE, "Positif"),
        ("neutral", COLOR_NEUTRAL, "Netral"),
        ("negative", COLOR_NEGATIVE, "Negatif"),
    ]:
        if col in df.columns:
            ax.plot(x, df[col].fillna(0), color=color, label=label,
                    linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Jumlah", fontsize=12)
    ax.set_title("Tren Sentimen Harian", fontsize=16, fontweight="bold",
                 pad=15, color=DARK_FG)
    ax.legend(loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.2)

    if "date" in df.columns:
        fig.autofmt_xdate()

    fig.tight_layout()
    logger.debug("Sentiment timeline chart rendered successfully.")
    return _fig_to_bytes(fig)


def render_topic_bar(topics: list[dict]) -> bytes:
    """
    Render horizontal bar chart topik utama berdasarkan frekuensi.

    Parameters
    ----------
    topics : list[dict]
        Setiap dict harus memiliki key 'topic_name' dan 'frequency'.

    Returns
    -------
    bytes
        PNG image bytes.
    """
    _apply_dark_theme()

    if not topics:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.text(0.5, 0.5, "Tidak ada data topik", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=DARK_FG)
        ax.set_title("Topik Utama", fontsize=16, fontweight="bold", color=DARK_FG)
        return _fig_to_bytes(fig)

    # Urutkan berdasarkan frekuensi
    sorted_topics = sorted(topics, key=lambda t: t.get("frequency", 0))
    names = [t.get("topic_name", "N/A") for t in sorted_topics]
    freqs = [t.get("frequency", 0) for t in sorted_topics]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax.barh(names, freqs, color="#3b82f6", height=0.6, edgecolor="none")

    for bar, val in zip(bars, freqs):
        ax.text(bar.get_width() + max(freqs) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=11, color=DARK_FG)

    ax.set_xlabel("Frekuensi", fontsize=12)
    ax.set_title("Topik Utama Percakapan", fontsize=16, fontweight="bold",
                 pad=15, color=DARK_FG)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    logger.debug("Topic bar chart rendered successfully.")
    return _fig_to_bytes(fig)


def render_engagement_bar(account_df: Any, n: int = 10) -> bytes:
    """
    Render horizontal bar chart top-N akun berdasarkan engagement.

    Parameters
    ----------
    account_df : pd.DataFrame | list[dict]
        Harus memiliki kolom 'akun' dan 'engagement'.
    n : int
        Jumlah akun teratas yang ditampilkan (default: 10).

    Returns
    -------
    bytes
        PNG image bytes.
    """
    import pandas as pd

    _apply_dark_theme()

    if account_df is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.text(0.5, 0.5, "Data akun tidak tersedia", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=DARK_FG)
        ax.set_title("Top Akun", fontsize=16, fontweight="bold", color=DARK_FG)
        return _fig_to_bytes(fig)

    if isinstance(account_df, list):
        account_df = pd.DataFrame(account_df)

    df = account_df.copy()

    # Cari kolom engagement
    eng_col = None
    for candidate in ["engagement", "engagement_score", "total_engagement", "influence_score"]:
        if candidate in df.columns:
            eng_col = candidate
            break
    if eng_col is None:
        eng_col = df.select_dtypes(include=[np.number]).columns[-1] if len(
            df.select_dtypes(include=[np.number]).columns) > 0 else None

    # Cari kolom akun
    akun_col = None
    for candidate in ["akun", "X akun", "account", "username"]:
        if candidate in df.columns:
            akun_col = candidate
            break
    if akun_col is None:
        akun_col = df.columns[0]

    if eng_col is None:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.text(0.5, 0.5, "Kolom engagement tidak ditemukan", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=DARK_FG)
        return _fig_to_bytes(fig)

    df = df.nlargest(n, eng_col)
    df = df.sort_values(eng_col)  # ascending untuk barh

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    bars = ax.barh(df[akun_col].astype(str), df[eng_col], color="#8b5cf6",
                   height=0.6, edgecolor="none")

    max_val = df[eng_col].max() if len(df) > 0 else 1
    for bar, val in zip(bars, df[eng_col]):
        ax.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}", va="center", fontsize=11, color=DARK_FG)

    ax.set_xlabel("Engagement", fontsize=12)
    ax.set_title(f"Top {n} Akun Berpengaruh", fontsize=16, fontweight="bold",
                 pad=15, color=DARK_FG)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    logger.debug("Engagement bar chart rendered successfully.")
    return _fig_to_bytes(fig)


def render_sarcasm_pie(sarcasm_count: int, non_sarcasm_count: int) -> bytes:
    """
    Render pie chart sarkasme vs non-sarkasme.

    Parameters
    ----------
    sarcasm_count : int
        Jumlah postingan sarkastik.
    non_sarcasm_count : int
        Jumlah postingan non-sarkastik.

    Returns
    -------
    bytes
        PNG image bytes.
    """
    _apply_dark_theme()

    sarcasm_count = max(0, int(sarcasm_count or 0))
    non_sarcasm_count = max(0, int(non_sarcasm_count or 0))

    if sarcasm_count + non_sarcasm_count == 0:
        sarcasm_count = non_sarcasm_count = 1  # hindari chart kosong

    sizes = [sarcasm_count, non_sarcasm_count]
    labels = ["Sarkasme", "Non-Sarkasme"]
    colors = ["#f97316", "#3b82f6"]
    explode = (0.05, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, explode=explode,
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 2},
        textprops={"fontsize": 13, "color": DARK_FG},
    )
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.set_title("Distribusi Sarkasme", fontsize=16, fontweight="bold",
                 pad=20, color=DARK_FG)

    logger.debug("Sarcasm pie chart rendered successfully.")
    return _fig_to_bytes(fig)


def render_emotion_trend(emotion_daily: list[dict] | Any) -> bytes:
    """
    Render stacked area chart tren emosi harian.

    Parameters
    ----------
    emotion_daily : list[dict] | pd.DataFrame
        Setiap entry: {"date": ..., "Marah": n, "Senang": n, ...}

    Returns
    -------
    bytes
        PNG image bytes.
    """
    import pandas as pd

    _apply_dark_theme()

    if emotion_daily is None or (isinstance(emotion_daily, list) and len(emotion_daily) == 0):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.text(0.5, 0.5, "Data tren emosi tidak tersedia", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=DARK_FG)
        ax.set_title("Tren Emosi Harian", fontsize=16, fontweight="bold", color=DARK_FG)
        return _fig_to_bytes(fig)

    if isinstance(emotion_daily, list):
        df = pd.DataFrame(emotion_daily)
    else:
        df = emotion_daily.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        x = df["date"]
    else:
        x = range(len(df))

    # Ambil kolom emosi yang tersedia
    emotion_cols = [c for c in EMOTION_COLORS if c in df.columns]
    if not emotion_cols:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.text(0.5, 0.5, "Kolom emosi tidak ditemukan", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=DARK_FG)
        return _fig_to_bytes(fig)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    y_stack = np.array([df[c].fillna(0).values for c in emotion_cols])
    colors_stack = [EMOTION_COLORS[c] for c in emotion_cols]

    ax.stackplot(x, *y_stack, labels=emotion_cols, colors=colors_stack, alpha=0.85)
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Jumlah", fontsize=12)
    ax.set_title("Tren Emosi Harian", fontsize=16, fontweight="bold",
                 pad=15, color=DARK_FG)
    ax.legend(loc="upper left", framealpha=0.8, fontsize=10)
    ax.grid(True, alpha=0.2)

    if "date" in df.columns:
        fig.autofmt_xdate()

    fig.tight_layout()
    logger.debug("Emotion trend chart rendered successfully.")
    return _fig_to_bytes(fig)


def render_wordcloud(keywords: list[tuple[str, int | float]]) -> bytes:
    """
    Render word cloud dari daftar keyword.

    Parameters
    ----------
    keywords : list[tuple[str, int|float]]
        Setiap tuple: (kata, frekuensi).

    Returns
    -------
    bytes
        PNG image bytes.
    """
    _apply_dark_theme()

    if not keywords:
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
        ax.text(0.5, 0.5, "Tidak ada keyword", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color=DARK_FG)
        ax.set_title("Word Cloud", fontsize=16, fontweight="bold", color=DARK_FG)
        ax.axis("off")
        return _fig_to_bytes(fig)

    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("Package 'wordcloud' tidak terinstall, "
                       "menampilkan bar chart sebagai fallback.")
        return _render_wordcloud_fallback(keywords)

    freq_dict = {str(word): float(freq) for word, freq in keywords if freq > 0}
    if not freq_dict:
        freq_dict = {"(kosong)": 1}

    wc = WordCloud(
        width=1500, height=900,
        background_color=DARK_BG,
        colormap="Set2",
        max_words=100,
        prefer_horizontal=0.7,
        min_font_size=10,
    ).generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud", fontsize=16, fontweight="bold",
                 pad=15, color=DARK_FG)
    fig.tight_layout()

    logger.debug("Word cloud rendered successfully.")
    return _fig_to_bytes(fig)


def _render_wordcloud_fallback(keywords: list[tuple[str, int | float]]) -> bytes:
    """Fallback bar chart jika wordcloud package tidak tersedia."""
    _apply_dark_theme()

    top = sorted(keywords, key=lambda x: x[1], reverse=True)[:20]
    top.reverse()  # ascending for barh
    words = [str(w) for w, _ in top]
    freqs = [float(f) for _, f in top]

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    ax.barh(words, freqs, color="#22d3ee", height=0.6)
    ax.set_xlabel("Frekuensi", fontsize=12)
    ax.set_title("Top Keywords", fontsize=16, fontweight="bold",
                 pad=15, color=DARK_FG)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    return _fig_to_bytes(fig)


def render_gauge(
    value: float,
    min_val: float = -100,
    max_val: float = 100,
) -> bytes:
    """
    Render gauge chart untuk Indeks Sentimen.

    Parameters
    ----------
    value : float
        Nilai indeks sentimen.
    min_val : float
        Nilai minimum gauge (default: -100).
    max_val : float
        Nilai maksimum gauge (default: 100).

    Returns
    -------
    bytes
        PNG image bytes.
    """
    _apply_dark_theme()

    value = _safe_pct(value, 0.0)
    value = max(min_val, min(max_val, value))  # clamp

    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": "polar"})

    # Gauge menggunakan setengah lingkaran (180°)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    # Buat segmen warna (merah → kuning → hijau)
    n_segments = 100
    theta_segments = np.linspace(np.pi, 0, n_segments)  # 180° ke 0° (kiri ke kanan)
    colors_gradient = []
    for i in range(n_segments):
        ratio = i / n_segments
        if ratio < 0.4:
            colors_gradient.append(COLOR_NEGATIVE)
        elif ratio < 0.6:
            colors_gradient.append("#eab308")  # kuning
        else:
            colors_gradient.append(COLOR_POSITIVE)

    for i in range(n_segments - 1):
        ax.barh(0.5, theta_segments[i] - theta_segments[i + 1],
                left=theta_segments[i + 1], height=0.35,
                color=colors_gradient[i], alpha=0.7)

    # Jarum penunjuk
    normalized = (value - min_val) / (max_val - min_val)  # 0..1
    needle_angle = np.pi * (1 - normalized)  # π → 0

    ax.plot([needle_angle, needle_angle], [0, 0.7], color="white",
            linewidth=3, zorder=10)
    ax.plot(needle_angle, 0.7, "o", color="white", markersize=8, zorder=10)
    ax.plot(needle_angle, 0, "o", color="white", markersize=6, zorder=10)

    # Label nilai
    ax.text(np.pi / 2, -0.25, f"{value:+.1f}",
            ha="center", va="center", fontsize=28, fontweight="bold",
            color=DARK_FG, transform=ax.transData)

    # Label min/max
    ax.text(np.pi, -0.05, f"{min_val}", ha="center", va="top",
            fontsize=11, color=DARK_FG)
    ax.text(0, -0.05, f"{max_val}", ha="center", va="top",
            fontsize=11, color=DARK_FG)

    fig.suptitle("Indeks Sentimen", fontsize=16, fontweight="bold",
                 color=DARK_FG, y=0.95)
    fig.tight_layout()

    logger.debug("Gauge chart rendered with value=%.2f.", value)
    return _fig_to_bytes(fig)
