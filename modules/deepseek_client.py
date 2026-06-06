"""
DeepSeek API Client untuk Analisis Sentimen
-------------------------------------------
Modul ini menangani komunikasi dengan DeepSeek API,
termasuk batching, retry, dan fallback ke lexicon-based.
"""
import os
import json
import time
import logging
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT = """Kamu adalah analis sentimen media sosial Indonesia yang sangat ahli.
Untuk setiap teks yang diberikan dalam array JSON, klasifikasikan dengan cermat.

Perhatikan:
- Bahasa gaul, slang, dan singkatan Indonesia (gw, lo, bngt, bgt, dll)
- Sarkasme dan ironi (misal: "bagus banget ya sampe error mulu" = NEGATIF)
- Emoji dan ekspresi informal
- Konteks kalimat, bukan hanya kata per kata

Output HARUS berupa JSON array dengan format PERSIS seperti ini (tanpa teks lain):
[
  {
    "sentiment": "Positive" | "Negative" | "Neutral",
    "emotion": "Marah" | "Takut" | "Jijik" | "Sedih" | "Antisipasi" | "Percaya" | "Terkejut" | "Senang" | "Netral",
    "confidence": 0.0 sampai 1.0,
    "reason": "alasan singkat dalam Bahasa Indonesia (max 10 kata)"
  }
]

Jumlah item output HARUS sama dengan jumlah teks input."""


def get_api_key() -> Optional[str]:
    """Ambil API key dari environment variable."""
    return os.environ.get("DEEPSEEK_API_KEY", "").strip() or None


def is_api_available() -> bool:
    """Cek apakah API key tersedia."""
    key = get_api_key()
    return bool(key and key.startswith("sk-") and len(key) > 10)


def analyze_batch(texts: list, api_key: str, retries: int = 2, timeout: int = 45) -> Optional[list]:
    """
    Kirim batch teks ke DeepSeek API dan kembalikan hasil analisis.

    Args:
        texts: List teks yang akan dianalisis (maks 20 per batch)
        api_key: DeepSeek API key
        retries: Jumlah maksimal percobaan ulang jika gagal
        timeout: Timeout dalam detik

    Returns:
        List dict hasil analisis, atau None jika gagal
    """
    if not texts:
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Buat user message berupa JSON array teks
    user_content = json.dumps(texts, ensure_ascii=False)

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analisis sentimen untuk {len(texts)} teks berikut:\n{user_content}"},
        ],
        "temperature": 0.1,   # Rendah agar konsisten
        "max_tokens": len(texts) * 80 + 200,  # Estimasi token output
        "response_format": {"type": "json_object"},
    }

    for attempt in range(retries + 1):
        try:
            logger.info(f"DeepSeek API: batch {len(texts)} teks (percobaan {attempt + 1})")
            resp = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if resp.status_code == 401:
                logger.error("DeepSeek API: API key tidak valid atau tidak aktif.")
                return None
            if resp.status_code == 429:
                logger.warning("DeepSeek API: Rate limit. Menunggu 5 detik...")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                logger.warning(f"DeepSeek API error {resp.status_code}: {resp.text[:200]}")
                if attempt < retries:
                    time.sleep(2)
                    continue
                return None

            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"].strip()

            # Parse JSON response
            parsed = _parse_response(raw_content, expected_count=len(texts))
            if parsed is not None:
                logger.info(f"DeepSeek API: berhasil menganalisis {len(parsed)} teks")
                return parsed
            else:
                logger.warning("DeepSeek API: format response tidak valid, mencoba ulang...")
                if attempt < retries:
                    time.sleep(1)
                    continue
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"DeepSeek API: timeout (percobaan {attempt + 1})")
            if attempt < retries:
                time.sleep(2)
        except requests.exceptions.ConnectionError:
            logger.warning(f"DeepSeek API: koneksi gagal (percobaan {attempt + 1})")
            if attempt < retries:
                time.sleep(3)
        except Exception as e:
            logger.error(f"DeepSeek API: error tidak terduga: {e}")
            if attempt < retries:
                time.sleep(1)

    return None


def _parse_response(content: str, expected_count: int) -> Optional[list]:
    """Parse dan validasi response JSON dari DeepSeek."""
    try:
        # Coba parse langsung
        data = json.loads(content)

        # Jika response dibungkus dalam object (karena response_format json_object)
        if isinstance(data, dict):
            # Cari key yang berisi array
            for key in data:
                if isinstance(data[key], list):
                    data = data[key]
                    break
            else:
                # Mungkin hanya satu item
                if all(k in data for k in ("sentiment", "emotion", "confidence")):
                    data = [data]

        if not isinstance(data, list):
            return None

        # Validasi dan normalisasi setiap item
        result = []
        valid_sentiments = {"Positive", "Negative", "Neutral"}
        valid_emotions = {"Marah", "Takut", "Jijik", "Sedih", "Antisipasi", "Percaya", "Terkejut", "Senang", "Netral"}

        for item in data:
            if not isinstance(item, dict):
                result.append(_default_result())
                continue

            sentiment = item.get("sentiment", "Neutral")
            if sentiment not in valid_sentiments:
                sentiment = "Neutral"

            emotion = item.get("emotion", "Netral")
            if emotion not in valid_emotions:
                emotion = "Netral"

            confidence = float(item.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            reason = str(item.get("reason", ""))[:100]

            result.append({
                "sentiment": sentiment,
                "emotion": emotion,
                "confidence": confidence,
                "reason": reason,
            })

        # Pastikan jumlah hasil sesuai dengan input
        while len(result) < expected_count:
            result.append(_default_result())

        return result[:expected_count]

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(f"Gagal parse response DeepSeek: {e}\nContent: {content[:300]}")
        return None


def _default_result() -> dict:
    """Hasil default jika parsing gagal."""
    return {
        "sentiment": "Neutral",
        "emotion": "Netral",
        "confidence": 0.3,
        "reason": "Tidak dapat dianalisis",
    }


def analyze_texts_with_deepseek(texts: list, api_key: str, batch_size: int = 15, max_workers: int = 5) -> list:
    """
    Analisis semua teks dengan DeepSeek API secara paralel menggunakan ThreadPoolExecutor.

    Args:
        texts: List semua teks yang akan dianalisis
        api_key: DeepSeek API key
        batch_size: Jumlah teks per batch (default 15 untuk stabilitas)
        max_workers: Jumlah thread paralel (default 5 untuk stabilitas rate limit)

    Returns:
        List hasil analisis (panjang sama dengan input texts)
    """
    if not texts:
        return []

    total = len(texts)
    batches = [texts[i:i + batch_size] for i in range(0, total, batch_size)]
    results_map = {}
    
    success_count = 0
    fallback_count = 0

    logger.info(f"Memulai analisis paralel dengan {len(batches)} batch dan {max_workers} worker threads.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches
        futures = {executor.submit(analyze_batch, batch, api_key): idx for idx, batch in enumerate(batches)}
        
        for future in as_completed(futures):
            idx = futures[future]
            batch_len = len(batches[idx])
            try:
                batch_result = future.result()
                if batch_result is not None and len(batch_result) == batch_len:
                    results_map[idx] = batch_result
                    success_count += batch_len
                else:
                    logger.warning(f"Batch {idx + 1} gagal atau tidak cocok, menggunakan fallback default.")
                    results_map[idx] = [_default_result()] * batch_len
                    fallback_count += batch_len
            except Exception as e:
                logger.error(f"Error pada thread batch {idx + 1}: {e}")
                results_map[idx] = [_default_result()] * batch_len
                fallback_count += batch_len

    # Gabungkan kembali sesuai urutan aslinya
    ordered_results = []
    for idx in range(len(batches)):
        ordered_results.extend(results_map[idx])

    logger.info(
        f"DeepSeek selesai: {success_count} via AI, "
        f"{fallback_count} via fallback dari total {total} teks (Paralel)"
    )
    return ordered_results
