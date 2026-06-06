import re
import math
from .preprocessor import clean_text, is_negation_word, is_intensifier

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
    "tolol", "konyol", "lucu", (  # lucu in negative sense
    "sayang",  # sayang as in "what a pity"
    ),
    "mubazir", "mubadzir", "boros",
    "krisis", "darurat", "genting", "kritis",
    "macet", "tersendat", "lamban", "lambat",
    "ambruk", "runtuh", "roboh", "tumbang",
    "curi", "rampok", "copet", "begal", "korupsi", "nyolong",
    "ambur", "amburadul", "berantakan", "semrawut",
    "nyebelin", "menyebalkan", "menggemaskan",  # menggemaskan as annoying
}

# Words that are negative in context but look positive - need special handling
CONTEXT_NEGATIVE = {
    "sayang",  # "sayang" as in "what a pity" vs "sayang" as in love
    "lucu",    # "lucu" as in "risible" vs "lucu" as funny
}

SENTIMENT_LEXICON = {}

for w in POSITIVE_WORDS:
    SENTIMENT_LEXICON[w] = 1
for w in NEGATIVE_WORDS:
    SENTIMENT_LEXICON[w] = -1

SENTIMENT_LEXICON["sayang"] = -1
SENTIMENT_LEXICON["lucu"] = 1


def score_text(text):
    if not text or not isinstance(text, str):
        return 0.0, 0.0, 0.0, "Neutral"
    words = text.lower().split()
    if not words:
        return 0.0, 0.0, 0.0, "Neutral"

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
            score += word_score
            word_count += 1
        elif is_negation_word(word):
            negation_active = True
            negation_span = 2

        if negation_span <= 0:
            negation_active = False

    if word_count == 0:
        return 0.0, 0.0, 0.0, "Neutral"

    avg_score = score / word_count
    normalized = max(-1, min(1, avg_score))

    if normalized > 0.15:
        label = "Positive"
    elif normalized < -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    confidence = min(1.0, abs(normalized) * 1.8)
    pos_score = max(0, normalized) if normalized > 0 else 0
    neg_score = max(0, -normalized) if normalized < 0 else 0
    neu_score = 1 - pos_score - neg_score

    return pos_score, neu_score, neg_score, label, confidence


def analyze_sentiment(df, text_column="clean_text"):
    results = df[text_column].apply(score_text)
    df["positive_score"] = results.apply(lambda x: x[0])
    df["neutral_score"] = results.apply(lambda x: x[1])
    df["negative_score"] = results.apply(lambda x: x[2])
    df["sentiment"] = results.apply(lambda x: x[3])
    df["confidence"] = results.apply(lambda x: x[4] if len(x) > 4 else 1.0)
    return df
