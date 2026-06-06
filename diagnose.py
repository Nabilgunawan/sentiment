"""Diagnosa import semua modul project"""
import sys
sys.path.insert(0, '.')

errors = []

# 1. dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] dotenv + .env loaded")
except Exception as e:
    print(f"[FAIL] dotenv: {e}")
    errors.append("dotenv")

# 2. flask
try:
    from flask import Flask
    print("[OK] flask")
except Exception as e:
    print(f"[FAIL] flask: {e}")
    errors.append("flask")

# 3. pandas
try:
    import pandas as pd
    print("[OK] pandas")
except Exception as e:
    print(f"[FAIL] pandas: {e}")
    errors.append("pandas")

# 4. preprocessor
try:
    from modules.preprocessor import preprocess_dataframe, clean_text
    print("[OK] modules.preprocessor")
except Exception as e:
    print(f"[FAIL] modules.preprocessor: {type(e).__name__}: {e}")
    errors.append("preprocessor")

# 5. sentiment
try:
    from modules.sentiment import analyze_sentiment
    print("[OK] modules.sentiment")
except Exception as e:
    print(f"[FAIL] modules.sentiment: {type(e).__name__}: {e}")
    errors.append("sentiment")

# 6. aggregator
try:
    from modules.aggregator import get_kpis, calculate_engagement
    print("[OK] modules.aggregator")
except Exception as e:
    print(f"[FAIL] modules.aggregator: {type(e).__name__}: {e}")
    errors.append("aggregator")

# 7. parser
try:
    from modules.parser import allowed_file, process_upload
    print("[OK] modules.parser")
except Exception as e:
    print(f"[FAIL] modules.parser: {type(e).__name__}: {e}")
    errors.append("parser")

# 8. deepseek_client
try:
    from modules.deepseek_client import is_api_available, get_api_key
    api_ok = is_api_available()
    key = get_api_key()
    masked = f"{key[:8]}...{key[-4:]}" if key else "tidak ada"
    print(f"[OK] modules.deepseek_client — API aktif: {api_ok}, key: {masked}")
except Exception as e:
    print(f"[FAIL] modules.deepseek_client: {type(e).__name__}: {e}")
    errors.append("deepseek_client")

# 9. summary
try:
    from modules.summary import generate_executive_summary
    print("[OK] modules.summary")
except Exception as e:
    print(f"[FAIL] modules.summary: {type(e).__name__}: {e}")
    errors.append("summary")

# 10. exporter
try:
    from modules.exporter import export_to_csv, export_to_json
    print("[OK] modules.exporter")
except Exception as e:
    print(f"[FAIL] modules.exporter: {type(e).__name__}: {e}")
    errors.append("exporter")

print()
if not errors:
    print("=" * 40)
    print("SEMUA OK — app.py siap dijalankan!")
    print("=" * 40)
else:
    print(f"Ada {len(errors)} modul bermasalah: {errors}")
    sys.exit(1)
