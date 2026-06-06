import re
import math
import os
import json
import logging
from typing import Optional
from .preprocessor import clean_text, is_negation_word, is_intensifier

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "baik", "bagus", "hebat", "mantap", "keren", "indah", "cantik", "menarik",
    "senang", "gembira", "bahagia", "suka", "cinta", "sayang", "suka",
    "puas", "bangga", "bersyukur", "semangat", "optimis", "cerah",
    "mudah", "nyaman", "enak", "lezat", "nikmat", "segar", "sehat",
    "bermanfaat", "membantu", "tepat", "akurat", "cepat", "handal",
    "berhasil", "sukses", "menang", "juara", "unggul", "terbaik",
    "luar biasa", "fantastis", "spektakuler", "istimewa", "sempurna",
    "ramah", "sopan", "baik hati", "dermawan", "jujur", "tulus", "ikhlas",
    "damai", "tenteram", "tenang", "harmonis", "rukun", "bersatu",
    "cerdas", "pintar", "brilian", "kreatif", "inovatif", "produktif",
    "maju", "berkembang", "modern", "canggih", "modern",
    "tegas", "adil", "berani", "jujur", "bijak", "bijaksana",
    "murah", "hemat", "efisien", "terjangkau", "gratis", "diskon",
    "ditunggu", "dinanti", "dirindukan", "populer", "favorit", "top",
    "setuju", "dukung", "support", "bantu", "amankan", "jaga",
    "recommended", "rekomendasi", "recomended", "recommend",
    "ok", "oke", "okey", "okelah", "baiklah", "sip", "siap",
    "wow", "amazing", "great", "nice", "good", "best", "perfect",
    "lengkap", "detail", "jelas", "transparan", "terbuka",
    "presisi", "terukur", "konsisten", "stabil", "kokoh", "kuat",
    "layak", "pantas", "patut", "wajar",
    "sembuh", "pulih", "membaik", "peningkatan", "perbaikan",
    "berkah", "barokah", "rahmat", "karunia", "nikmat",
    "lancar", "mulus", "gampang", "praktis", "simpel",
    "meriah", "ramai", "viral", "hits", "booming",
    "faedah", "guna", "manfaat", "kontribusi", "dedikasi",
    "prestasi", "pencapaian", "kemajuan", "kesuksesan",
}

NEGATIVE_WORDS = {
    "buruk", "jelek", "parah", "menyedihkan", "mengerikan", "menjijikkan",
    "sedih", "kecewa", "kesal", "jengkel", "marah", "benci", "sebal",
    "gagal", "rugi", "kalah", "bodoh", "tolol", "bego", "dungu",
    "susah", "sulit", "payah", "berat", "rumit", "ribet", "repot",
    "lambat", "lemot", "lemah", "jelek", "rendah", "turun",
    "kurang", "minim", "sedikit", "terbatas", "habis", "kosong",
    "mahal", "boros", "mubazir", "sia-sia", "percuma",
    "sakit", "cedera", "luka", "lelah", "letih", "lesu", "pusing",
    "sial", "celaka", "bencana", "musibah", "malapetaka",
    "krisis", "resesi", "inflasi", "korupsi", "kriminal", "kejahatan",
    "konflik", "perang", "bertengkar", "ribut", "gaduh", "kacau",
    "sampah", "kotor", "jorok", "kumuh", "bau",
    "menipu", "bohong", "curang", "khianat", "korupsi",
    "hina", "caci", "makian", "fitnah", "ghibah", "gosip",
    "ancam", "teror", "intimidasi", "tekanan",
    "mati", "tewas", "meninggal", "musnah", "hancur", "rusak",
    "larang", "tolak", "cegah", "halang", "hambat",
    "menyesal", "sesal", "nyesel", "sesali",
    "aneh", "asing", "ganjil", "janggal", "nyeleneh",
    "mumet", "pusing", "stress", "stres", "depresi", "cemas", "gelisah",
    "takut", "khawatir", "waswas", "cemas",
    "dosa", "neraka", "azab", "siksa", "laknat", "kutuk",
    "tolol", "konyol", "lucu",
    "sayang",
    "mubazir", "mubadzir", "boros",
    "krisis", "darurat", "genting", "kritis",
    "macet", "tersendat", "lamban", "lambat",
    "ambruk", "runtuh", "roboh", "tumbang",
    "curi", "rampok", "copet", "begal", "korupsi", "nyolong",
    "ambur", "amburadul", "berantakan", "semrawut",
    "nyebelin", "menyebalkan", "menggemaskan",
    "error", "bug", "crash", "force close", "lemot", "ngadat",
    "kecewa", "kecewa banget", "keeeewa", "kece",
}

SARCASM_PATTERNS = [
    (r"\bbagus\s+banget\b.*\b(?:error|gagal|rusak|tidak|nggak|ga\s+bisa|lemot|down)\b", "Negative"),
    (r"\bkeren\b.*\b(?:gagal|error|nggak|tidak|rusak)\b", "Negative"),
    (r"\bmantap\b.*\b(?:error|gagal|tidak|nggak)\b", "Negative"),
    (r"\bhebat\b.*\b(?:rusak|gagal|error|tidak|nggak)\b", "Negative"),
    (r"\bsukses\b.*\b(?:gagal|error|nggak|tidak)\b", "Negative"),
    (r"\bkeren\s+banget\b.*\b(?:sampai|kok|tapi)\b", "Negative"),
    (r"\bsangat\s+(?:baik|bagus|hebat)\b.*\b(?:tetapi|tapi|namun)\b", "Negative"),
    (r"\b(?:tentu|pasti|tentulah)\s+(?:sangat|sekali)\b.*\b(?:error|gagal)\b", "Negative"),
    (r"\bterima\s+kasih\b.*\b(?:error|rusak|gagal|lemot)\b", "Negative"),
    (r"\b(?:mantap|keren|bagus).*?(?:tapi|tetapi|sayangnya|namun)", "Mixed"),
    (r"\bgampang\b.*\b(?:error|gagal|nggak)\b", "Negative"),
    (r"\b(?:wow|waah)\b.*\b(?:parah|jelek|buruk|payah)\b", "Negative"),
    (r"\bmemang\s+(?:bagus|hebat|keren)\b.*\b(?:tapi|tetapi)\b", "Negative"),
]

SENTIMENT_LEXICON = {}

for w in POSITIVE_WORDS:
    SENTIMENT_LEXICON[w] = 1
for w in NEGATIVE_WORDS:
    SENTIMENT_LEXICON[w] = -1

SENTIMENT_LEXICON["sayang"] = -1
SENTIMENT_LEXICON["lucu"] = 1


def detect_sarcasm(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    lower = text.lower()
    for pattern, sentiment in SARCASM_PATTERNS:
        if re.search(pattern, lower):
            return sentiment
    return None


def score_text(text):
    if not text or not isinstance(text, str):
        return 0.0, 0.0, 0.0, "Neutral", 0.5

    sarcasm_result = detect_sarcasm(text)
    if sarcasm_result == "Negative":
        return 0.0, 0.0, 1.0, "Negative", 0.85
    elif sarcasm_result == "Mixed":
        pass

    words = text.lower().split()
    if not words:
        return 0.0, 0.0, 0.0, "Neutral", 0.5

    score = 0.0
    word_count = 0
    negation_active = False
    negation_span = 0

    for i, word in enumerate(words):
        if word in SENTIMENT_LEXICON:
            word_score = SENTIMENT_LEXICON[word]
            if is_intensifier(word):
                continue
            if negation_active and negation_span > 0:
                word_score *= -0.5
                negation_span -= 1
                if negation_span == 0:
                    negation_active = False
            elif negation_active:
                word_score *= -1
                negation_span -= 1
                if negation_span == 0:
                    negation_active = False
            if i > 0 and is_intensifier(words[i - 1]):
                word_score *= 1.5
            # Check for exclamation marks
            if word.endswith("!") or word.endswith("!!"):
                word_score *= 1.3
            score += word_score
            word_count += 1
        elif is_negation_word(word):
            negation_active = True
            negation_span = 2

        if negation_span <= 0:
            negation_active = False

    if word_count == 0:
        return 0.0, 0.0, 0.0, "Neutral", 0.5

    avg_score = score / word_count
    normalized = max(-1, min(1, avg_score))

    if normalized > 0.15:
        label = "Positive"
    elif normalized < -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    # Better confidence: based on absolute magnitude and word count
    magnitude = abs(normalized)
    length_factor = min(1.0, word_count / 8)
    confidence = min(1.0, magnitude * 1.5 + length_factor * 0.2)
    if label == "Neutral":
        confidence = max(0.3, min(0.7, confidence))

    pos_score = max(0, normalized) if normalized > 0 else 0
    neg_score = max(0, -normalized) if normalized < 0 else 0
    neu_score = 1 - pos_score - neg_score

    return pos_score, neu_score, neg_score, label, round(confidence, 4)


def analyze_with_deepseek(texts, api_key, batch_size=15):
    from .deepseek_client import analyze_texts_with_deepseek
    try:
        results = analyze_texts_with_deepseek(texts, api_key, batch_size)
        return results
    except Exception as e:
        logger.warning(f"DeepSeek AI gagal: {e}")
        return None


def analyze_sentiment(df, text_column="clean_text", api_key=None):
    api_available = False
    if api_key and api_key.startswith("sk-") and len(api_key) > 10:
        api_available = True

    if api_available:
        try:
            from .deepseek_client import is_api_available
            if is_api_available():
                texts = df[text_column].fillna("").tolist()
                ai_results = analyze_with_deepseek(texts, api_key)
                if ai_results and len(ai_results) == len(texts):
                    df["sentiment"] = [r["sentiment"] for r in ai_results]
                    df["emotion"] = [r["emotion"] for r in ai_results]
                    df["confidence"] = [r["confidence"] for r in ai_results]
                    df["ai_reason"] = [r["reason"] for r in ai_results]
                    df["positive_score"] = (df["sentiment"] == "Positive").astype(float)
                    df["negative_score"] = (df["sentiment"] == "Negative").astype(float)
                    df["neutral_score"] = (df["sentiment"] == "Neutral").astype(float)
                    logger.info(f"Analisis AI berhasil untuk {len(texts)} teks")
                    return df
        except Exception as e:
            logger.warning(f"AI analysis failed, falling back to lexicon: {e}")

    logger.info("Menggunakan lexicon-based sentiment analysis (fallback)")
    results = df[text_column].apply(score_text)
    df["positive_score"] = results.apply(lambda x: x[0])
    df["neutral_score"] = results.apply(lambda x: x[1])
    df["negative_score"] = results.apply(lambda x: x[2])
    df["sentiment"] = results.apply(lambda x: x[3])
    df["confidence"] = results.apply(lambda x: x[4] if len(x) > 4 else 1.0)
    return df
