"""
services/deepseek_client.py
----------------------------
DeepSeek API Client — HANYA untuk Insight Generation & Executive Summary.
TIDAK digunakan untuk klasifikasi sentimen (sudah digantikan IndoBERT).

Menerima hasil akhir analisis (sentiment + emotion + sarcasm + topics)
dan menghasilkan laporan naratif profesional dalam Bahasa Indonesia.
"""
import os
import json
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import config
    DEEPSEEK_API_URL = config.DEEPSEEK_API_URL
    DEEPSEEK_MODEL = config.DEEPSEEK_MODEL
    DEEPSEEK_API_KEY = config.DEEPSEEK_API_KEY
except ImportError:
    DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
    DEEPSEEK_MODEL = "deepseek-chat"
    DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()

# ── System Prompt untuk Narrative Analysis ────────────────────────────────────
SUMMARY_SYSTEM_PROMPT = """Kamu adalah analis senior media sosial dan komunikasi strategis Indonesia.

Tugasmu adalah menulis LAPORAN EKSEKUTIF profesional berdasarkan data analisis sentimen yang diberikan.

Gaya penulisan:
- Profesional, formal, dan analitis
- Bahasa Indonesia baku yang baik dan benar
- Menggunakan data kuantitatif untuk mendukung setiap klaim
- Menyertakan insight strategis dan rekomendasi kebijakan
- Siap digunakan untuk laporan pemerintah, stakeholder, atau manajemen

Format output HARUS dalam Markdown dengan heading yang jelas.

Struktur laporan:
1. Gambaran Umum (ringkasan singkat temuan utama)
2. Temuan Sentimen (distribusi, tren, anomali)
3. Analisis Emosi (emosi dominan, implikasi)
4. Deteksi Sarkasme (pola sarkasme, contoh, implikasi)
5. Topik Utama (topik terpopuler, korelasi dengan sentimen)
6. Analisis Spike (lonjakan sentimen, penyebab potensial)
7. Akun Berpengaruh (influencer utama, peran mereka)
8. Konten Viral (karakteristik konten viral)
9. Insight Strategis (3-5 insight kunci)
10. Rekomendasi Kebijakan (3-5 rekomendasi konkret)
11. Kesimpulan

Panjang: 2000-3000 kata. Tulis dalam satu output tanpa jeda."""


def get_api_key() -> Optional[str]:
    """Ambil API key dari environment variable."""
    key = DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    return key if key else None


def is_api_available() -> bool:
    """Cek apakah API key tersedia dan valid."""
    key = get_api_key()
    return bool(key and key.startswith("sk-") and len(key) > 10)


def generate_ai_summary(analysis_data: dict, retries: int = 1, timeout: int = 45) -> Optional[str]:
    """
    Generate executive summary menggunakan DeepSeek AI.

    Args:
        analysis_data: Dict berisi semua hasil analisis:
            - kpis: KPI metrics
            - sentiment_index: float
            - emotion_distribution: dict
            - sarcasm_distribution: dict
            - topics: list of topics
            - sentiment_spikes: list
            - viral_posts: list
            - accounts: list of top accounts
            - keywords: list of top keywords
            - bigrams: list
            - trigrams: list
            - date_range: str
            - total_rows: int
            - analysis_mode: str
        retries: Jumlah maksimal percobaan ulang
        timeout: Timeout dalam detik

    Returns:
        String executive summary dalam Markdown, atau None jika gagal
    """
    api_key = get_api_key()
    if not api_key:
        logger.warning("DeepSeek API key tidak tersedia untuk summary generation")
        return None

    # Format data analisis sebagai prompt
    user_content = _format_analysis_prompt(analysis_data)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }

    for attempt in range(retries + 1):
        try:
            logger.info(f"DeepSeek Summary: percobaan {attempt + 1}/{retries + 1}")
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if resp.status_code == 401:
                logger.error("DeepSeek API: API key tidak valid")
                return None
            if resp.status_code == 429:
                logger.warning("DeepSeek API: Rate limit, menunggu 5 detik...")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                logger.warning(f"DeepSeek API error {resp.status_code}: {resp.text[:200]}")
                if attempt < retries:
                    time.sleep(2)
                    continue
                return None

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            logger.info(f"DeepSeek Summary berhasil: {len(content)} karakter")
            return content

        except requests.exceptions.Timeout:
            logger.warning(f"DeepSeek API timeout (percobaan {attempt + 1})")
            if attempt < retries:
                time.sleep(2)
        except requests.exceptions.ConnectionError:
            logger.warning(f"DeepSeek API koneksi gagal (percobaan {attempt + 1})")
            if attempt < retries:
                time.sleep(3)
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            if attempt < retries:
                time.sleep(1)

    return None


def _format_analysis_prompt(data: dict) -> str:
    """Format data analisis menjadi prompt untuk DeepSeek."""
    parts = ["Berikut adalah hasil analisis sentimen media sosial yang perlu kamu buatkan laporan eksekutif:\n"]

    # KPIs
    kpis = data.get("kpis", {})
    parts.append("## Data KPI:")
    parts.append(f"- Total Post: {kpis.get('total_post', 0):,}")
    parts.append(f"- Akun Unik: {kpis.get('unique_accounts', 0):,}")
    parts.append(f"- Positif: {kpis.get('positive', 0):,} ({kpis.get('positive_pct', 0)}%)")
    parts.append(f"- Netral: {kpis.get('neutral', 0):,} ({kpis.get('neutral_pct', 0)}%)")
    parts.append(f"- Negatif: {kpis.get('negative', 0):,} ({kpis.get('negative_pct', 0)}%)")
    parts.append(f"- Total Engagement: {kpis.get('total_engagement', 0):,}")
    parts.append(f"- Total Views: {kpis.get('total_views', 0):,}")
    parts.append(f"- Sentiment Index: {data.get('sentiment_index', 0)}")
    parts.append(f"- Sarcasm: {kpis.get('sarcasm_count', 0):,} ({kpis.get('sarcasm_pct', 0)}%)")
    parts.append(f"- Periode: {data.get('date_range', 'N/A')}")
    parts.append(f"- Metode: IndoBERT Transformer + {data.get('analysis_mode', 'N/A')}")

    # Emotion Distribution
    emotion_dist = data.get("emotion_distribution", {})
    if emotion_dist:
        parts.append("\n## Distribusi Emosi:")
        for em, info in emotion_dist.items():
            if isinstance(info, dict):
                parts.append(f"- {em}: {info.get('count', 0)} ({info.get('percentage', 0)}%)")
            else:
                parts.append(f"- {em}: {info}")

    # Sarcasm Distribution
    sarcasm_dist = data.get("sarcasm_distribution", {})
    if sarcasm_dist:
        parts.append("\n## Distribusi Sarkasme:")
        parts.append(f"- Sarkastik: {sarcasm_dist.get('sarcastic', 0)}")
        parts.append(f"- Non-sarkastik: {sarcasm_dist.get('non_sarcastic', 0)}")

    # Topics
    topics = data.get("topics", [])
    if topics:
        parts.append("\n## Topik Utama:")
        for t in topics[:10]:
            parts.append(f"- {t.get('name', 'N/A')}: {t.get('frequency', 0)} post")

    # Spikes
    spikes = data.get("sentiment_spikes", [])
    if spikes:
        parts.append("\n## Lonjakan Sentimen:")
        for s in spikes[:5]:
            parts.append(f"- {s.get('date', 'N/A')}: {s.get('type', 'N/A')} (magnitude: {s.get('magnitude', 0)}σ)")

    # Top Accounts
    accounts = data.get("accounts", [])
    if accounts:
        parts.append("\n## Akun Terpopuler:")
        for a in accounts[:5]:
            if isinstance(a, dict):
                parts.append(
                    f"- {a.get('X akun', 'N/A')}: {a.get('total_post', 0)} post, "
                    f"{a.get('total_engagement', 0):,} engagement"
                )

    # Viral Posts
    viral = data.get("viral_posts", [])
    if viral:
        parts.append("\n## Post Viral:")
        for v in viral[:5]:
            konten = v.get("konten", "")[:100]
            parts.append(
                f"- [{v.get('sentiment', 'N/A')}] {v.get('akun', 'N/A')}: "
                f"\"{konten}\" (engagement: {v.get('engagement_score', 0):.0f})"
            )

    # Keywords
    keywords = data.get("keywords", [])
    if keywords:
        kw_str = ", ".join(
            f"{k.get('word', k) if isinstance(k, dict) else k[0]}"
            for k in keywords[:15]
        )
        parts.append(f"\n## Keywords Utama: {kw_str}")

    parts.append("\n\nBuatkan laporan eksekutif profesional berdasarkan data di atas.")
    return "\n".join(parts)
