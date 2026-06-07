"""
preprocessing/normalizer.py
---------------------------
Modul normalisasi Bahasa Indonesia: bahasa gaul, singkatan, typo umum.

Kamus ini digunakan untuk menormalkan teks media sosial Indonesia
sebelum dikirim ke model Transformer. Normalisasi ini penting karena
Transformer di-train dengan bahasa Indonesia formal/semi-formal,
sehingga bahasa gaul perlu diterjemahkan terlebih dahulu.
"""
import re
import logging

logger = logging.getLogger(__name__)

# ── Kamus Bahasa Gaul Indonesia ──────────────────────────────────────────────
# Mapping bahasa gaul/slang → bahasa baku
SLANG_DICT: dict[str, str] = {
    # Negasi
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak",
    "kagak": "tidak", "tdk": "tidak", "kgk": "tidak", "ngga": "tidak",
    "ngg": "tidak", "g": "tidak", "enggak": "tidak", "engga": "tidak",
    "gx": "tidak", "kaga": "tidak", "ogah": "tidak mau",
    "gpp": "tidak apa apa", "gapapa": "tidak apa apa",

    # Singkatan umum
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "dlm": "dalam",
    "dr": "dari", "krn": "karena", "tp": "tapi", "ttg": "tentang",
    "jd": "jadi", "lg": "lagi", "blm": "belum", "sdh": "sudah",
    "skrg": "sekarang", "bs": "bisa", "bgt": "banget", "bngt": "banget",
    "spt": "seperti", "sm": "sama", "pd": "pada", "trs": "terus",
    "dg": "dengan", "dmn": "dimana", "gmn": "gimana", "knp": "kenapa",
    "kpn": "kapan", "brp": "berapa", "spy": "supaya", "hrs": "harus",
    "bt": "buat", "bkn": "bukan", "emg": "memang", "emng": "memang",
    "mmg": "memang", "stlh": "setelah", "sblm": "sebelum",
    "krg": "kurang", "lbh": "lebih", "blg": "bilang",
    "ngmng": "ngomong", "tgl": "tanggal", "bln": "bulan",
    "thn": "tahun", "org": "orang", "tmn": "teman",
    "msh": "masih", "klo": "kalau", "kl": "kalau", "klu": "kalau",

    # Kata ganti
    "gw": "saya", "gue": "saya", "gua": "saya", "ane": "saya",
    "w": "saya", "sy": "saya", "ak": "saya", "aku": "saya",
    "lo": "kamu", "lu": "kamu", "elu": "kamu", "loe": "kamu",
    "u": "kamu", "km": "kamu", "nt": "kamu",

    # Ekspresi gaul
    "mantul": "mantap betul", "mantap": "mantap", "mantab": "mantap",
    "mantep": "mantap", "mntp": "mantap",
    "kece": "keren", "keren": "keren", "krn": "karena",
    "cuy": "", "bro": "", "sis": "", "gan": "", "kak": "",
    "wkwk": "haha", "wkwkwk": "haha", "wkwkwkwk": "haha",
    "kwkw": "haha", "kwkwkw": "haha",
    "hehe": "haha", "hihi": "haha", "huhu": "sedih",
    "anjir": "anjing", "anjg": "anjing", "anjr": "anjing",
    "bgst": "bangsat", "bngst": "bangsat",
    "kampret": "sialan", "kmprt": "sialan",
    "asw": "asli", "aslii": "asli",
    "bgmn": "bagaimana", "gmna": "gimana",
    "bgmna": "bagaimana",

    # Intensifier gaul
    "bgt": "banget", "bngt": "banget", "bngtt": "banget",
    "bgtt": "banget", "bangett": "banget",
    "bener": "benar", "bnr": "benar",
    "pol": "sekali", "pisan": "sekali",
    "parah": "parah", "prh": "parah",
    "gilak": "gila", "gilaa": "gila",

    # Positif gaul
    "josss": "bagus sekali", "joss": "bagus sekali",
    "gg": "hebat", "pro": "profesional",
    "sat set": "cepat", "satset": "cepat",
    "gacor": "bagus", "sultan": "kaya",
    "sabi": "bisa", "sbi": "bisa",
    "cuan": "untung", "stonks": "untung",
    "receh": "lucu", "ngakak": "lucu sekali",
    "wkwk": "haha", "halu": "halusinasi",

    # Negatif gaul
    "zonk": "gagal", "ambyar": "hancur",
    "gabut": "tidak ada kerjaan", "mager": "malas gerak",
    "baper": "bawa perasaan", "kesel": "kesal",
    "gedeg": "kesal", "gemes": "gemas",
    "capek": "capai", "cape": "capai", "cpk": "capai",
    "males": "malas", "mls": "malas",
    "ribet": "rumit", "rbt": "rumit",
    "lebay": "berlebihan", "lby": "berlebihan",
    "toxic": "beracun", "cringe": "memalukan",

    # Afirmasi
    "iya": "iya", "iy": "iya", "yoi": "iya", "yup": "iya",
    "yap": "iya", "yep": "iya", "iyes": "iya",
    "sip": "baik", "siap": "baik", "ok": "baik", "oke": "baik",
    "okey": "baik", "okee": "baik",

    # Kata kerja gaul
    "nyimak": "menyimak", "nonton": "menonton",
    "nyari": "mencari", "nyoba": "mencoba",
    "ngapain": "sedang apa", "ngapa": "kenapa",
    "ntar": "nanti", "ntr": "nanti", "tar": "nanti",
    "btw": "omong omong", "fyi": "untuk informasi",
    "otw": "dalam perjalanan", "asap": "secepatnya",

    # Typo umum
    "mslh": "masalah", "msalah": "masalah",
    "pntng": "penting", "pnting": "penting",
    "pmerintah": "pemerintah", "pmrintah": "pemerintah",
    "rakyat": "rakyat", "rkyt": "rakyat",
    "korupsi": "korupsi", "korup": "korupsi",
    "demokrasi": "demokrasi", "dmkrasi": "demokrasi",
    "ekonomi": "ekonomi", "eknomi": "ekonomi",
    "pendidikan": "pendidikan", "pndidikan": "pendidikan",
    "kesehatan": "kesehatan", "kshatan": "kesehatan",
    "lingkungan": "lingkungan", "lngkungan": "lingkungan",

    # Media sosial
    "repost": "repost", "rp": "repost",
    "dm": "pesan langsung", "pm": "pesan pribadi",
    "fyp": "for you page", "viral": "viral",
    "trending": "trending", "tt": "trending topic",
    "follower": "pengikut", "following": "mengikuti",
    "like": "suka", "share": "bagikan",
    "comment": "komentar", "subscribe": "berlangganan",
}

# ── Kamus Singkatan Formal ──────────────────────────────────────────────────
ABBREVIATION_DICT: dict[str, str] = {
    "dpr": "dewan perwakilan rakyat",
    "dprd": "dewan perwakilan rakyat daerah",
    "mpr": "majelis permusyawaratan rakyat",
    "mk": "mahkamah konstitusi",
    "ma": "mahkamah agung",
    "kpk": "komisi pemberantasan korupsi",
    "bnpb": "badan nasional penanggulangan bencana",
    "bpjs": "badan penyelenggara jaminan sosial",
    "umkm": "usaha mikro kecil menengah",
    "apbn": "anggaran pendapatan belanja negara",
    "apbd": "anggaran pendapatan belanja daerah",
    "ppkm": "pemberlakuan pembatasan kegiatan masyarakat",
    "psbb": "pembatasan sosial berskala besar",
    "uu": "undang undang",
    "pp": "peraturan pemerintah",
    "perpres": "peraturan presiden",
    "permen": "peraturan menteri",
    "perda": "peraturan daerah",
    "polri": "kepolisian republik indonesia",
    "tni": "tentara nasional indonesia",
    "asn": "aparatur sipil negara",
    "pns": "pegawai negeri sipil",
    "bbm": "bahan bakar minyak",
    "pln": "perusahaan listrik negara",
    "bumn": "badan usaha milik negara",
    "bumd": "badan usaha milik daerah",
    "ihsg": "indeks harga saham gabungan",
    "ojk": "otoritas jasa keuangan",
    "bi": "bank indonesia",
    "ri": "republik indonesia",
    "nkri": "negara kesatuan republik indonesia",
    "pilkada": "pemilihan kepala daerah",
    "pilpres": "pemilihan presiden",
    "pileg": "pemilihan legislatif",
    "pemilu": "pemilihan umum",
    "kpu": "komisi pemilihan umum",
    "bawaslu": "badan pengawas pemilihan umum",
    "ormas": "organisasi masyarakat",
    "lsm": "lembaga swadaya masyarakat",
    "ham": "hak asasi manusia",
    "kdrt": "kekerasan dalam rumah tangga",
    "phk": "pemutusan hubungan kerja",
    "ump": "upah minimum provinsi",
    "umr": "upah minimum regional",
    "esg": "environmental social governance",
    "sdm": "sumber daya manusia",
    "sda": "sumber daya alam",
    "amdal": "analisis mengenai dampak lingkungan",
    "rtrw": "rencana tata ruang wilayah",
}


def normalize_text(text: str) -> str:
    """
    Normalisasi teks Bahasa Indonesia: ubah slang, singkatan, dan typo ke bentuk baku.

    Args:
        text: Teks yang sudah dibersihkan (lowercase, tanpa URL/mention)

    Returns:
        Teks yang sudah dinormalisasi
    """
    if not text or not isinstance(text, str):
        return ""

    words = text.split()
    normalized = []

    for word in words:
        word_clean = word.strip()
        if not word_clean:
            continue

        # Cek di kamus slang
        if word_clean in SLANG_DICT:
            replacement = SLANG_DICT[word_clean]
            if replacement:  # Skip jika replacement kosong (filler words)
                normalized.append(replacement)
            continue

        # Cek di kamus singkatan formal
        if word_clean in ABBREVIATION_DICT:
            normalized.append(ABBREVIATION_DICT[word_clean])
            continue

        # Tidak ditemukan di kamus, pertahankan kata asli
        normalized.append(word_clean)

    return " ".join(normalized)


def normalize_batch(texts: list[str]) -> list[str]:
    """
    Normalisasi batch teks sekaligus.

    Args:
        texts: List teks yang sudah dibersihkan

    Returns:
        List teks yang sudah dinormalisasi
    """
    return [normalize_text(t) for t in texts]
