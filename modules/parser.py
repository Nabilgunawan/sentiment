import re
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"csv", "xlsx", "html", "htm"}
REQUIRED_COLUMNS = {"Tanggal", "Waktu", "X akun", "Konten"}
OPTIONAL_COLUMNS = {"Komentar", "Repost", "Likes", "Views", "Link"}
COLUMN_ALIASES = {
    "tanggal": "Tanggal", "date": "Tanggal", "tgl": "Tanggal",
    "waktu": "Waktu", "time": "Waktu", "jam": "Waktu",
    "akun": "X akun", "username": "X akun", "user": "X akun", "account": "X akun", "author": "X akun",
    "content": "Konten", "teks": "Konten", "tweet": "Konten", "text": "Konten", "post": "Konten", "caption": "Konten",
    "comment": "Komentar", "comments": "Komentar", "komentar": "Komentar",
    "repost": "Repost", "retweet": "Repost", "share": "Repost", "rt": "Repost",
    "like": "Likes", "likes": "Likes", "suka": "Likes",
    "view": "Views", "views": "Views", "dilihat": "Views", "impression": "Views",
    "url": "Link", "link": "Link", "links": "Link",
}
NUMERIC_MAP = {
    "rb": 1000, "ribu": 1000, "k": 1000,
    "jt": 1000000, "juta": 1000000, "m": 1000000,
    "miliar": 1000000000, "b": 1000000000,
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    renamed = {}
    for col in df.columns:
        key = col.lower().strip()
        if key in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key]
        elif key.endswith("s") and key[:-1] in COLUMN_ALIASES:
            renamed[col] = COLUMN_ALIASES[key[:-1]]
    if renamed:
        df = df.rename(columns=renamed)
    return df


def validate_columns(df):
    df = normalize_columns(df)
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom wajib tidak ditemukan: {', '.join(missing)}. "
            f"Kolom yang tersedia: {', '.join(df.columns)}. "
            f"Pastikan file memiliki kolom: {', '.join(REQUIRED_COLUMNS)}"
        )
    return df


def parse_number(text):
    if pd.isna(text):
        return 0
    if isinstance(text, (int, float)):
        return int(text)
    s = str(text).strip().lower().replace(" ", "").replace(",", ".").replace("\xa0", "")
    if s == "" or s == "-":
        return 0
    for suffix, multiplier in sorted(NUMERIC_MAP.items(), key=lambda x: -len(x[0])):
        if suffix in s:
            num_part = s.replace(suffix, "").strip()
            try:
                return int(float(num_part) * multiplier)
            except ValueError:
                pass
    try:
        return int(float(s))
    except ValueError:
        return 0


def parse_datetime(date_str, time_str):
    if pd.isna(date_str) or pd.isna(time_str):
        return pd.NaT
    date_str = str(date_str).strip()
    time_str = str(time_str).strip()
    combined = f"{date_str} {time_str}"
    fmts = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d %b %Y %H:%M",
        "%d %B %Y %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    try:
        return pd.to_datetime(combined, errors="coerce")
    except Exception:
        return pd.NaT


def parse_html_table(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    tables = soup.find_all("table")
    if not tables:
        tables = soup.find_all("tbody")
    for table in tables:
        df = pd.read_html(str(table))[0]
        if len(df.columns) >= 4:
            return df
    raise ValueError("Tidak dapat menemukan tabel HTML yang valid di file.")


def parse_file(filepath, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "csv":
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                df = pd.read_csv(filepath, encoding=enc)
                break
            except (UnicodeDecodeError, Exception):
                continue
        else:
            df = pd.read_csv(filepath, encoding="utf-8", encoding_errors="replace")
    elif ext == "xlsx":
        df = pd.read_excel(filepath, engine="openpyxl")
    elif ext in ("html", "htm"):
        df = parse_html_table(filepath)
    else:
        raise ValueError(f"Format file {ext} tidak didukung.")
    return df


def deduplicate(df):
    link_col = "Link" if "Link" in df.columns else None
    if link_col and df[link_col].notna().sum() > 0:
        before = len(df)
        df = df.drop_duplicates(subset=[link_col], keep="first")
        if len(df) < before:
            return df, before - len(df)
    key_cols = ["datetime", "X akun", "Konten"]
    available = [c for c in key_cols if c in df.columns]
    if len(available) >= 2:
        before = len(df)
        df = df.drop_duplicates(subset=available, keep="first")
        return df, before - len(df)
    return df, 0


def process_upload(filepath, filename):
    df = parse_file(filepath, filename)
    df = validate_columns(df)
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df["datetime"] = df.apply(
        lambda r: parse_datetime(r["Tanggal"], r["Waktu"]), axis=1
    )
    for col in ["Komentar", "Repost", "Likes", "Views"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_number)
    df, dup_count = deduplicate(df)
    return df, dup_count
