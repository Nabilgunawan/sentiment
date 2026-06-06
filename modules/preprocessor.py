import re
import math


STOPWORDS_ID = {
    "dan", "di", "ke", "dari", "yang", "dengan", "ini", "itu", "untuk", "pada",
    "adalah", "akan", "telah", "sudah", "bisa", "dapat", "ada", "tidak", "juga",
    "atau", "saya", "kamu", "dia", "kami", "kita", "mereka", "anda", "aku",
    "oleh", "sebagai", "dalam", "dalamnya", "sebuah", "satu", "saat", "setelah",
    "sebelum", "antara", "tentang", "seperti", "secara", "semua", "hanya",
    "karena", "jika", "maka", "bahwa", "bukan", "hal", "lebih", "masih",
    "saja", "lain", "baru", "sangat", "besar", "sedang", "seluruh", "serta",
    "lagi", "pun", "tetapi", "namun", "sedangkan", "melainkan", "sementara",
    "sehingga", "makanya", "maka", "yakni", "yaitu", "apalagi", "lagi-lagi",
    "begitu", "begini", "begitulah", "begitupun", "meski", "meskipun",
    "walaupun", "kendati", "biar", "biarpun", "walau", "sekalipun",
    "sebab", "karenanya", "olehkarenaitu", "oleh sebab itu",
    "sungguh", "benar", "benar-benar", "pasti", "tentu", "tentunya",
    "paling", "terlalu", "begitu", "sekali", "sangat", "amat",
    "agak", "cukup", "kurang", "sedikit", "banyak", "beberapa",
    "semua", "seluruh", "setiap", "masing-masing", "para",
    "lainnya", "lain-lain", "sesuatu", "suatu", "macam", "jenis",
    "ini", "itu", "sini", "situ", "sono", "sana",
    "seperti", "bagai", "bagaikan", "seolah", "seolah-olah",
    "seakan", "seakan-akan", "laksana", "ibarat",
    "saya", "aku", "kami", "kita", "engkau", "kau", "dikau",
    "kamu", "anda", "saudara", "saudari",
    "ia", "dia", "beliau", "mereka",
    "buah", "ekor", "orang", "batang", "biji", "helai",
    "lembar", "utas", "carik", "potong", "kuntum", "pucuk",
    "kerat", "keping", "kepal", "pihak", "sosok", "tokoh",
    "si", "sang", "para", "kaum", "umat", "bangsa",
    "ada", "ialah", "merupakan", "yakni", "yaitu",
    "bahwa", "apabila", "bila", "bilamana",
    "saat", "tatkala", "kala", "ketika", "sewaktu",
    "selagi", "selama", "semenjak", "sejak", "selamanya",
    "hingga", "sampai", "sampai-sampai",
    "i", "ni", "tu", "ko", "mu", "nyo", "nyi",
    "to", "dong", "sih", "deh", "lah", "kah", "pun",
    "juga", "pula", "terus", "malah", "bahkan", "apalagi",
    "jangan", "janganlah", "tanpa", "supaya", "biar", "biarlah",
    "ayo", "mari", "silakan", "harap", "mohon",
    "tolong", "coba", "cobalah", "hendak", "hendaknya",
    "sebaiknya", "seharusnya", "semestinya", "mestinya",
    "boleh", "bolehkan", "perbolehkan", "izinkan",
    "ya", "oh", "wah", "aduh", "nah", "hmm", "hai", "halo",
    "assalamualaikum", "salam", "maaf", "terima kasih", "trimakasih",
    "percaya", "rasa", "pikir", "anggap", "kira",
    "mau", "ingin", "hendak", "berniat", "bermaksud",
}

NEGATION_WORDS = {
    "tidak", "bukan", "tak", "nggak", "gak", "ga", "tdk",
    "belum", "jangan", "tanpa", "kagak", "bkn",
}

INTENSIFIERS = {
    "sangat", "amat", "sungguh", "benar-benar", "paling",
    "sekali", "terlalu", "begitu", "luar biasa", "banget",
    "super", "ekstra", "betul-betul", "betul",
}


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    emoji_pattern = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF\U0000200D\U0000FE0F\U0000200D"
        "\U00002600-\U000026FF\U00002700-\U000027BF]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    words = text.split()
    return " ".join(w for w in words if w not in STOPWORDS_ID)


def is_negation_word(word):
    return word in NEGATION_WORDS


def is_intensifier(word):
    return word in INTENSIFIERS


def simple_stem(word):
    word = word.strip()
    if len(word) <= 3:
        return word
    suffixes = ["kan", "nya", "kah", "lah", "pun", "ku", "mu", "i"]
    prefixes = [
        ("ber", 3), ("be", 2), ("me", 2), ("mem", 3), ("men", 3),
        ("meng", 4), ("meny", 4), ("per", 3), ("pe", 2),
        ("pem", 3), ("pen", 3), ("peng", 4), ("peny", 4),
        ("ter", 3), ("te", 2), ("ke", 2), ("se", 2),
        ("di", 2), ("meng", 4), ("meny", 4),
    ]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    for prefix, length in prefixes:
        if word.startswith(prefix) and len(word) > length + 2:
            word = word[length:]
            if word.startswith("l") and len(word) > 2:
                pass
            break
    if word in ("nyari", "nyatu", "nyoba", "nyanyi"):
        return word[1:]
    if word.startswith("ny") and len(word) > 3:
        word = word[1:]
    if word.startswith("ng") and len(word) > 3:
        word = word[2:]
    if word.startswith("nge") and len(word) > 5:
        word = word[3:]
    word = word.replace("ny", "c", 1) if word.startswith("nyc") else word
    if word in ("makan", "minum", "tidur", "jalan", "baca", "tulis"):
        pass
    return word if word else word


def preprocess(text, do_stemming=False):
    cleaned = clean_text(text)
    cleaned = remove_stopwords(cleaned)
    if do_stemming:
        words = cleaned.split()
        stemmed = [simple_stem(w) for w in words]
        cleaned = " ".join(stemmed)
    return cleaned


def preprocess_dataframe(df, text_column="Konten", do_stemming=False):
    df = df.copy()
    if text_column in df.columns:
        df["clean_text"] = df[text_column].apply(
            lambda x: preprocess(x, do_stemming)
        )
    return df


import pandas as pd
