"""
services/topic_service.py
--------------------------
Layanan pemodelan topik dan ekstraksi kata kunci dari teks Bahasa Indonesia.
Menggunakan BERTopic untuk topic modeling dan KeyBERT untuk keyword extraction.

Fitur utama:
- BERTopic dengan sentence-transformers embedding untuk topic modeling
- KeyBERT untuk ekstraksi kata kunci berbasis semantik
- Fallback TF-based jika BERTopic/KeyBERT tidak terinstall
- Konfigurasi melalui config.py (embedding model, min topic size, dll)
- Penanganan graceful saat dependency berat tidak tersedia
"""

import re
import logging
import math
from collections import Counter
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)

# ── Cek ketersediaan library opsional ────────────────────────────────────────

_BERTOPIC_AVAILABLE: bool = False
_KEYBERT_AVAILABLE: bool = False
_SENTENCE_TRANSFORMERS_AVAILABLE: bool = False

try:
    from bertopic import BERTopic  # type: ignore[import-untyped]
    _BERTOPIC_AVAILABLE = True
except ImportError:
    logger.info("BERTopic tidak terinstall. Menggunakan fallback TF-based.")

try:
    from keybert import KeyBERT  # type: ignore[import-untyped]
    _KEYBERT_AVAILABLE = True
except ImportError:
    logger.info("KeyBERT tidak terinstall. Keyword extraction menggunakan fallback.")

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.info("sentence-transformers tidak terinstall.")

# ── Indonesian Stopwords (subset untuk fallback TF) ─────────────────────────

_INDONESIAN_STOPWORDS: set[str] = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "tidak", "akan", "juga", "sudah", "ada", "dalam",
    "bisa", "atau", "bukan", "saya", "aku", "kamu", "dia", "mereka",
    "kita", "kami", "nya", "lah", "kan", "ya", "dong", "deh", "sih",
    "nih", "tuh", "lo", "gue", "lu", "gw", "yg", "dgn", "utk", "dlm",
    "tp", "tdk", "sdh", "blm", "jgn", "krn", "trs", "bgt", "sm",
    "se", "ber", "ter", "per", "me", "meng", "men", "mem", "meny",
    "ke", "di", "pen", "pem", "peng", "peny", "an", "kan", "lah",
    "sangat", "sekali", "lebih", "paling", "terlalu", "cukup",
    "hanya", "saja", "masih", "telah", "belum", "sedang", "seperti",
    "oleh", "antara", "setiap", "semua", "banyak", "beberapa",
    "lagi", "terus", "tetapi", "tapi", "namun", "maka", "jika",
    "kalau", "ketika", "saat", "waktu", "agar", "supaya", "karena",
    "sebab", "jadi", "maka", "meski", "walau", "apakah", "siapa",
    "apa", "dimana", "kapan", "bagaimana", "mengapa", "kenapa",
    "bila", "hingga", "sampai", "sebelum", "sesudah", "selama",
    "mau", "biar", "dulu", "nanti", "lalu", "kemudian", "baru",
    "sering", "jarang", "pernah", "hampir", "ingin", "hendak",
    "perlu", "harus", "boleh", "dapat", "mampu", "wajib",
    "sebuah", "suatu", "para", "begitu", "demikian", "tersebut",
}


class TopicService:
    """
    Layanan pemodelan topik dan ekstraksi kata kunci.

    Menggunakan BERTopic + sentence-transformers untuk topic modeling
    dan KeyBERT untuk keyword extraction. Menyediakan fallback berbasis
    term frequency jika library berat tidak tersedia.

    Attributes:
        topic_model: Instance BERTopic (atau None jika tidak tersedia)
        keyword_model: Instance KeyBERT (atau None jika tidak tersedia)
        embedding_model: Nama/path model embedding
    """

    def __init__(self) -> None:
        """Inisialisasi TopicService dengan penanganan dependency opsional."""
        self.topic_model: Any = None
        self.keyword_model: Any = None
        self.embedding_model_name: str = config.TOPIC_MODEL_EMBEDDING
        self._use_fallback_topics: bool = False
        self._use_fallback_keywords: bool = False

        self._initialize_models()

    # ── Inisialisasi Model ───────────────────────────────────────────────

    def _initialize_models(self) -> None:
        """Inisialisasi BERTopic dan KeyBERT jika tersedia."""
        if not config.USE_TRANSFORMERS:
            logger.info("USE_TRANSFORMERS set to False. Menggunakan fallback TF/TF-IDF secara instan untuk pemodelan topik.")
            self._use_fallback_topics = True
            self._use_fallback_keywords = True
            return

        # Inisialisasi BERTopic
        if _BERTOPIC_AVAILABLE and _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Resolve nr_topics
                nr_topics = config.TOPIC_NR_TOPICS
                if isinstance(nr_topics, str) and nr_topics.lower() == "auto":
                    nr_topics_param: Any = "auto"
                else:
                    try:
                        nr_topics_param = int(nr_topics)
                    except (ValueError, TypeError):
                        nr_topics_param = "auto"

                embedding_model = SentenceTransformer(self.embedding_model_name)

                self.topic_model = BERTopic(
                    embedding_model=embedding_model,
                    min_topic_size=config.TOPIC_MIN_TOPIC_SIZE,
                    nr_topics=nr_topics_param,
                    top_n_words=config.TOPIC_TOP_N_WORDS,
                    verbose=False,
                )
                logger.info(
                    f"BERTopic diinisialisasi dengan embedding: "
                    f"{self.embedding_model_name}"
                )
            except Exception as e:
                logger.warning(f"Gagal menginisialisasi BERTopic: {e}")
                self._use_fallback_topics = True
        else:
            self._use_fallback_topics = True
            if not _BERTOPIC_AVAILABLE:
                logger.info("BERTopic tidak tersedia, menggunakan fallback TF-based.")
            if not _SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info(
                    "sentence-transformers tidak tersedia, "
                    "topic modeling menggunakan fallback."
                )

        # Inisialisasi KeyBERT
        if _KEYBERT_AVAILABLE:
            try:
                self.keyword_model = KeyBERT(model=self.embedding_model_name)
                logger.info("KeyBERT diinisialisasi.")
            except Exception as e:
                logger.warning(f"Gagal menginisialisasi KeyBERT: {e}")
                self._use_fallback_keywords = True
        else:
            self._use_fallback_keywords = True
            logger.info(
                "KeyBERT tidak tersedia, keyword extraction menggunakan fallback."
            )

    # ── Ekstraksi Topik ──────────────────────────────────────────────────

    def extract_topics(
        self,
        texts: list[str],
        n_topics: int = 10,
    ) -> dict:
        """
        Ekstraksi topik dari kumpulan teks.

        Args:
            texts: List teks yang akan dianalisis
            n_topics: Jumlah maksimum topik yang diinginkan

        Returns:
            Dictionary berisi:
                - topics (list[dict]): Daftar topik, masing-masing berisi
                    name, frequency, representative_posts
                - topic_over_time (list[dict]): Distribusi topik (kosong
                    jika data temporal tidak tersedia)
        """
        if not texts:
            return {"topics": [], "topic_over_time": []}

        # Filter teks kosong
        valid_texts = [
            t for t in texts if t and isinstance(t, str) and t.strip()
        ]
        if not valid_texts:
            return {"topics": [], "topic_over_time": []}

        if self._use_fallback_topics or self.topic_model is None:
            return self._fallback_extract(valid_texts, n_topics)

        try:
            return self._bertopic_extract(valid_texts, n_topics)
        except Exception as e:
            logger.error(
                f"Error pada BERTopic, menggunakan fallback: {e}"
            )
            return self._fallback_extract(valid_texts, n_topics)

    def _bertopic_extract(
        self, texts: list[str], n_topics: int
    ) -> dict:
        """Ekstraksi topik menggunakan BERTopic."""
        # BERTopic membutuhkan minimal beberapa dokumen
        if len(texts) < config.TOPIC_MIN_TOPIC_SIZE:
            logger.warning(
                f"Jumlah teks ({len(texts)}) kurang dari min_topic_size "
                f"({config.TOPIC_MIN_TOPIC_SIZE}). Menggunakan fallback."
            )
            return self._fallback_extract(texts, n_topics)

        topics_list, probs = self.topic_model.fit_transform(texts)

        # Ambil info topik dari BERTopic
        topic_info = self.topic_model.get_topic_info()
        result_topics: list[dict] = []

        for _, row in topic_info.iterrows():
            topic_id = row.get("Topic", -1)
            # Skip outlier topic (-1)
            if topic_id == -1:
                continue

            topic_words = self.topic_model.get_topic(topic_id)
            if topic_words:
                topic_name = "_".join([w for w, _ in topic_words[:3]])
            else:
                topic_name = f"Topik_{topic_id}"

            # Kumpulkan representative posts
            topic_indices = [
                i for i, t in enumerate(topics_list) if t == topic_id
            ]
            representative_posts = [
                texts[i] for i in topic_indices[:5]
            ]

            result_topics.append({
                "name": topic_name,
                "frequency": int(row.get("Count", len(topic_indices))),
                "representative_posts": representative_posts,
            })

            if len(result_topics) >= n_topics:
                break

        # Sort berdasarkan frequency (tertinggi duluan)
        result_topics.sort(key=lambda x: x["frequency"], reverse=True)

        return {
            "topics": result_topics,
            "topic_over_time": [],  # Data temporal memerlukan kolom tanggal
        }

    # ── Ekstraksi Kata Kunci ─────────────────────────────────────────────

    def extract_keywords(
        self,
        texts: list[str],
        n_keywords: int = 10,
    ) -> list[dict]:
        """
        Ekstraksi kata kunci dari kumpulan teks.

        Args:
            texts: List teks yang akan dianalisis
            n_keywords: Jumlah kata kunci yang diinginkan

        Returns:
            List dictionary, masing-masing berisi:
                - keyword (str): Kata kunci
                - score (float): Skor relevansi (0.0-1.0)
        """
        if not texts:
            return []

        valid_texts = [
            t for t in texts if t and isinstance(t, str) and t.strip()
        ]
        if not valid_texts:
            return []

        if self._use_fallback_keywords or self.keyword_model is None:
            return self._fallback_extract_keywords(valid_texts, n_keywords)

        try:
            return self._keybert_extract(valid_texts, n_keywords)
        except Exception as e:
            logger.error(
                f"Error pada KeyBERT, menggunakan fallback: {e}"
            )
            return self._fallback_extract_keywords(valid_texts, n_keywords)

    def _keybert_extract(
        self, texts: list[str], n_keywords: int
    ) -> list[dict]:
        """Ekstraksi kata kunci menggunakan KeyBERT."""
        # Gabungkan semua teks menjadi satu dokumen untuk ekstraksi global
        combined_text = " ".join(texts)

        keywords = self.keyword_model.extract_keywords(
            combined_text,
            keyphrase_ngram_range=(1, 2),
            stop_words=list(_INDONESIAN_STOPWORDS),
            top_n=n_keywords,
            use_maxsum=True,
            nr_candidates=min(n_keywords * 3, 50),
        )

        return [
            {"keyword": kw, "score": round(float(score), 4)}
            for kw, score in keywords
        ]

    # ── Fallback TF-Based ────────────────────────────────────────────────

    def _fallback_extract(
        self, texts: list[str], n_topics: int = 10
    ) -> dict:
        """
        Ekstraksi topik sederhana berbasis Term Frequency sebagai fallback.

        Menghitung frekuensi n-gram (unigram dan bigram) dari seluruh teks,
        menghapus stopwords, lalu mengelompokkan kata-kata paling sering
        sebagai 'topik'.

        Args:
            texts: List teks yang akan dianalisis
            n_topics: Jumlah topik yang diinginkan

        Returns:
            Dictionary dengan format yang sama dengan BERTopic output
        """
        if not texts:
            return {"topics": [], "topic_over_time": []}

        # Tokenisasi dan hitung frekuensi
        word_counter: Counter[str] = Counter()
        bigram_counter: Counter[str] = Counter()
        text_word_map: dict[str, list[int]] = {}  # kata → indeks teks

        for idx, text in enumerate(texts):
            words = self._tokenize(text)
            for word in words:
                word_counter[word] += 1
                if word not in text_word_map:
                    text_word_map[word] = []
                text_word_map[word].append(idx)

            # Bigrams
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                bigram_counter[bigram] += 1

        # Gabungkan unigram dan bigram, prioritaskan bigram
        combined: Counter[str] = Counter()
        for word, count in word_counter.items():
            combined[word] = count
        for bigram, count in bigram_counter.items():
            if count >= 2:  # Bigram minimal muncul 2 kali
                combined[bigram] = count * 2  # Beri bobot lebih pada bigram

        # Ambil topik teratas
        top_terms = combined.most_common(n_topics * 3)

        result_topics: list[dict] = []
        seen_words: set[str] = set()

        for term, frequency in top_terms:
            if len(result_topics) >= n_topics:
                break

            # Hindari duplikasi (jika unigram sudah ada sebagai bagian bigram)
            term_words = set(term.split())
            if term_words & seen_words and len(term_words) == 1:
                continue
            seen_words.update(term_words)

            # Kumpulkan representative posts
            representative_posts: list[str] = []
            first_word = term.split()[0]
            if first_word in text_word_map:
                post_indices = text_word_map[first_word][:5]
                representative_posts = [texts[i] for i in post_indices]

            result_topics.append({
                "name": term,
                "frequency": frequency,
                "representative_posts": representative_posts,
            })

        return {
            "topics": result_topics,
            "topic_over_time": [],
        }

    def _fallback_extract_keywords(
        self, texts: list[str], n_keywords: int = 10
    ) -> list[dict]:
        """
        Ekstraksi kata kunci berbasis TF-IDF sederhana sebagai fallback.

        Args:
            texts: List teks yang akan dianalisis
            n_keywords: Jumlah kata kunci yang diinginkan

        Returns:
            List dictionary berisi keyword dan score
        """
        if not texts:
            return []

        # Hitung TF dan DF
        tf: Counter[str] = Counter()
        df: Counter[str] = Counter()
        n_docs = len(texts)

        for text in texts:
            words = self._tokenize(text)
            tf.update(words)
            unique_words = set(words)
            df.update(unique_words)

        # Hitung TF-IDF
        tfidf_scores: dict[str, float] = {}
        for word, term_freq in tf.items():
            doc_freq = df.get(word, 1)
            idf = math.log((n_docs + 1) / (doc_freq + 1)) + 1
            tfidf_scores[word] = term_freq * idf

        # Normalisasi skor ke 0-1
        if tfidf_scores:
            max_score = max(tfidf_scores.values())
            if max_score > 0:
                tfidf_scores = {
                    k: v / max_score for k, v in tfidf_scores.items()
                }

        # Ambil top N
        sorted_keywords = sorted(
            tfidf_scores.items(), key=lambda x: x[1], reverse=True
        )[:n_keywords]

        return [
            {"keyword": kw, "score": round(score, 4)}
            for kw, score in sorted_keywords
        ]

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenisasi teks sederhana dengan filtering stopwords.

        Args:
            text: Teks yang akan ditokenisasi

        Returns:
            List token yang sudah difilter
        """
        if not text or not isinstance(text, str):
            return []

        # Lowercase dan hapus karakter non-alfanumerik
        text_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        words = text_clean.split()

        # Filter: hapus stopwords, kata terlalu pendek, dan angka murni
        filtered = [
            w for w in words
            if w not in _INDONESIAN_STOPWORDS
            and len(w) > 2
            and not w.isdigit()
        ]
        return filtered

    # ── Helpers ──────────────────────────────────────────────────────────

    def is_bertopic_available(self) -> bool:
        """Cek apakah BERTopic berhasil diinisialisasi."""
        return self.topic_model is not None and not self._use_fallback_topics

    def is_keybert_available(self) -> bool:
        """Cek apakah KeyBERT berhasil diinisialisasi."""
        return self.keyword_model is not None and not self._use_fallback_keywords

    def __repr__(self) -> str:
        topic_mode = "BERTopic" if self.is_bertopic_available() else "TF Fallback"
        kw_mode = "KeyBERT" if self.is_keybert_available() else "TF-IDF Fallback"
        return f"<TopicService topics={topic_mode} keywords={kw_mode}>"
