"""
modules/emotion.py
------------------
Modul deteksi emosi Bahasa Indonesia berbasis lexicon Plutchik (8 emosi dasar).
Digunakan sebagai fallback atau pelengkap analisis DeepSeek AI.
"""

from collections import defaultdict

# ── Lexicon 8 Emosi (Plutchik Wheel) ────────────────────────────────────────

EMOTION_LEXICON = {
    "Marah": {
        "marah", "kesal", "geram", "murka", "emosi", "berang", "jengkel",
        "sebal", "benci", "dongkol", "sengit", "gusar", "meradang", "naik darah",
        "sewot", "gondok", "mangkel", "nyebelin", "nyebel", "ngeselin",
        "menyebalkan", "menggemaskan", "geregetan", "sebal banget",
        "kesel", "nyolot", "arogan", "sombong", "kurang ajar", "lancang",
        "brengsek", "bajingan", "kampret", "bangsat", "sialan",
    },
    "Takut": {
        "takut", "khawatir", "cemas", "panik", "gelisah", "waswas", "ngeri",
        "horor", "trauma", "fobia", "gentar", "ciut", "grogi", "gugup",
        "deg-degan", "was-was", "resah", "risau", "gamang", "galau",
        "tegang", "merinding", "bergidik", "ketakutan", "paranoid",
        "pesimis", "putus asa", "hopeless",
    },
    "Jijik": {
        "jijik", "muak", "mual", "benci", "memuakkan", "menjijikkan",
        "kotor", "jorok", "najis", "busuk", "bau", "menjijikan",
        "vulgar", "cabul", "mesum", "amoral", "brengsek", "sampah",
        "laknat", "terkutuk", "nista", "hina", "rendah",
    },
    "Sedih": {
        "sedih", "pilu", "menangis", "duka", "nelangsa", "muram", "kecewa",
        "patah hati", "galau", "merana", "sendu", "murung", "gundah",
        "trenyuh", "haru", "terharu", "iba", "kasihan", "miris",
        "menyedihkan", "pedih", "perih", "sakit hati", "tersiksa",
        "menderita", "nestapa", "malang", "sial", "celaka",
        "kehilangan", "rindu", "kangen", "savvy", "hancur",
    },
    "Antisipasi": {
        "menunggu", "berharap", "ingin", "nantikan", "semoga",
        "mudah-mudahan", "insya allah", "doakan", "harap", "minta",
        "perlu", "butuh", "mau", "hendak", "rencana", "akan",
        "siap", "persiapan", "ancang-ancang", "antisipasi", "waspada",
        "hati-hati", "awas", "jaga", "lindungi", "waspadai",
    },
    "Percaya": {
        "yakin", "percaya", "optimis", "mantap", "pasti", "andal",
        "terpercaya", "amanah", "handal", "reliable", "terbukti",
        "akurat", "valid", "sahih", "benar", "tepat", "sesuai",
        "konsisten", "stabil", "kokoh", "solid", "kuat", "tangguh",
        "dukungan", "dukung", "support", "setuju", "sepakat",
        "bangga", "apresiasi", "salut", "kagum", "hormat",
    },
    "Terkejut": {
        "kaget", "terkejut", "tidak menyangka", "shock", "wow",
        "astaga", "seriusan", "gila", "gokil", "luar biasa",
        "tidak percaya", "mustahil", "tidak terduga", "mendadak",
        "tiba-tiba", "mengejutkan", "spektakuler", "fantastis",
        "dahsyat", "luar biasa", "amazing", "incredible", "unbelievable",
        "astagfirullah", "subhanallah", "masya allah", "ajaib",
    },
    "Senang": {
        "senang", "bahagia", "gembira", "suka", "ceria", "riang",
        "antusias", "excited", "girang", "sukacita", "bersyukur",
        "alhamdulillah", "terima kasih", "makasih", "thx", "thanks",
        "mantap", "keren", "bagus", "bagus banget", "sip", "siap",
        "oke", "ok", "good", "nice", "great", "best", "top",
        "recommended", "rekomendasi", "puas", "lega", "tenang",
        "damai", "harmonis", "indah", "cantik", "hebat", "wow",
        "lucu", "seru", "asik", "menyenangkan", "menggemaskan",
    },
}

# Bobot per emosi untuk normalisasi
EMOTION_WEIGHTS = {
    "Marah": 1.2,
    "Takut": 1.1,
    "Jijik": 1.1,
    "Sedih": 1.0,
    "Antisipasi": 0.8,
    "Percaya": 0.9,
    "Terkejut": 0.9,
    "Senang": 1.0,
}


class EmotionDetector:
    """Detektor emosi Bahasa Indonesia berbasis lexicon."""

    def __init__(self):
        # Buat index terbalik: kata → list emosi
        self._word_index: dict[str, list[str]] = defaultdict(list)
        for emotion, words in EMOTION_LEXICON.items():
            for word in words:
                self._word_index[word.lower()].append(emotion)

    def detect(self, text: str) -> dict[str, float]:
        """
        Deteksi skor tiap emosi dari teks.

        Returns:
            Dict {nama_emosi: skor 0.0–1.0}
        """
        if not text or not isinstance(text, str):
            return {e: 0.0 for e in EMOTION_LEXICON}

        words = text.lower().split()
        raw_scores: dict[str, float] = defaultdict(float)
        word_count = max(len(words), 1)

        for word in words:
            if word in self._word_index:
                for emotion in self._word_index[word]:
                    raw_scores[emotion] += EMOTION_WEIGHTS.get(emotion, 1.0)

        # Normalisasi 0–1 berdasarkan panjang teks
        result = {}
        for emotion in EMOTION_LEXICON:
            score = raw_scores.get(emotion, 0.0) / word_count
            result[emotion] = min(1.0, round(score * 5, 4))  # scale up agar lebih readable

        return result

    def get_dominant_emotion(self, text: str) -> tuple[str, float]:
        """
        Return emosi dominan beserta skornya.

        Returns:
            Tuple (nama_emosi, skor)
        """
        scores = self.detect(text)
        if not any(scores.values()):
            return ("Netral", 0.0)
        dominant = max(scores, key=scores.get)
        return (dominant, scores[dominant])

    def detect_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Proses list teks sekaligus."""
        return [self.detect(t) for t in texts]


# Singleton instance untuk digunakan di seluruh aplikasi
detector = EmotionDetector()


def detect_emotion(text: str) -> dict[str, float]:
    """Helper function untuk deteksi emosi satu teks."""
    return detector.detect(text)


def get_dominant_emotion(text: str) -> str:
    """Helper function untuk mendapatkan emosi dominan."""
    name, _ = detector.get_dominant_emotion(text)
    return name


def aggregate_emotions(emotion_list: list[dict]) -> dict[str, float]:
    """
    Agregasi distribusi emosi dari list hasil deteksi.

    Returns:
        Dict {emosi: rata_rata_skor}
    """
    if not emotion_list:
        return {e: 0.0 for e in EMOTION_LEXICON}

    totals = defaultdict(float)
    for em in emotion_list:
        for emotion, score in em.items():
            totals[emotion] += score

    n = len(emotion_list)
    return {e: round(totals[e] / n, 4) for e in EMOTION_LEXICON}
