"""
preprocessing/cleaner.py
------------------------
Modul pembersihan teks Bahasa Indonesia untuk model Transformer.
Menghapus URL, mention, hashtag symbol, karakter berulang, emoji, dan HTML tags.

PENTING: Tidak melakukan stemming, stopword removal, atau TF-IDF
karena Transformer model menangani fitur-fitur ini secara internal.
"""
import re
import html
import logging

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Membersihkan teks media sosial untuk input model Transformer.

    Pipeline:
    1. Unescape HTML entities
    2. Hapus HTML tags
    3. Hapus URL
    4. Hapus mention (@username)
    5. Hapus hashtag symbol tapi pertahankan kata (#BBMNaik → BBMNaik)
    6. Hapus karakter berulang berlebihan (baguuuus → baguus)
    7. Hapus emoji yang tidak relevan
    8. Case folding ke lowercase
    9. Normalisasi whitespace

    Args:
        text: Teks mentah dari media sosial

    Returns:
        Teks yang sudah dibersihkan
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Unescape HTML entities (&amp; &lt; &gt; dll)
    text = html.unescape(text)

    # 2. Hapus HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Hapus URL (http, https, www, t.co, dll)
    text = re.sub(
        r"https?://\S+|www\.\S+|t\.co/\S+|bit\.ly/\S+|goo\.gl/\S+",
        " ", text
    )

    # 4. Hapus mention (@username)
    text = re.sub(r"@\w+", " ", text)

    # 5. Hapus hashtag symbol tapi pertahankan kata
    # #BBMNaik → BBMNaik, #korupsi → korupsi
    text = re.sub(r"#(\w+)", r"\1", text)

    # 6. Hapus emoji yang tidak relevan (mempertahankan beberapa emoji sentimen)
    # Hapus seluruh blok emoji Unicode
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"   # Emoticons
        "\U0001F300-\U0001F5FF"   # Misc Symbols & Pictographs
        "\U0001F680-\U0001F6FF"   # Transport & Map
        "\U0001F1E0-\U0001F1FF"   # Flags
        "\U0001F900-\U0001F9FF"   # Supplemental Symbols
        "\U0001FA00-\U0001FA6F"   # Chess Symbols
        "\U0001FA70-\U0001FAFF"   # Symbols Extended-A
        "\U00002702-\U000027B0"   # Dingbats
        "\U000024C2-\U0001F251"
        "\U0000200D"              # Zero Width Joiner
        "\U0000FE0F"              # Variation Selector
        "\U00002600-\U000026FF"   # Misc symbols
        "\U00002700-\U000027BF"   # Dingbats
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(" ", text)

    # 7. Hapus karakter berulang berlebihan (3+ karakter berulang → 2)
    # baguuuuus → baguus, hahahahaha → haha
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 8. Case folding
    text = text.lower()

    # 9. Hapus karakter non-alfanumerik kecuali spasi dan tanda hubung
    # Pertahankan huruf, angka, spasi, dan tanda hubung (untuk kata serapan)
    text = re.sub(r"[^\w\s\-]", " ", text)

    # 10. Normalisasi whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_batch(texts: list[str]) -> list[str]:
    """
    Bersihkan batch teks sekaligus.

    Args:
        texts: List teks mentah

    Returns:
        List teks yang sudah dibersihkan
    """
    return [clean_text(t) for t in texts]
