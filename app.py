import os
import uuid
import json
import pandas as pd
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import (
    Flask, render_template, request, jsonify,
    send_file, session, redirect, url_for, Response,
)
from werkzeug.utils import secure_filename
from modules.parser import allowed_file, process_upload
from modules.preprocessor import preprocess_dataframe
from modules.sentiment import analyze_sentiment
from modules.aggregator import (
    calculate_engagement, calculate_engagement_weighted,
    aggregate_daily, aggregate_by_account, aggregate_by_emotion,
    extract_top_keywords, extract_bigrams, extract_trigrams,
    detect_sentiment_spikes, get_top_viral_posts, get_kpis,
    get_sentiment_index, engagement_weighted_sentiment,
)
from modules.summary import generate_executive_summary
from modules.exporter import export_to_csv, export_to_json
from modules.deepseek_client import get_api_key, is_api_available

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32).hex())
app.config["UPLOAD_FOLDER"] = os.environ.get(
    "UPLOAD_FOLDER",
    os.path.join(os.path.dirname(__file__), "uploads"),
)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Format file tidak didukung. Gunakan: .csv, .xlsx, atau .html"
        }), 400

    session_id = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{session_id}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(filepath)

    try:
        do_stemming = request.form.get("stemming", "0") == "1"
        df, dup_count = process_upload(filepath, file.filename)
        df = calculate_engagement(df)
        df = calculate_engagement_weighted(df)
        df = preprocess_dataframe(df, text_column="Konten", do_stemming=do_stemming)

        api_key = get_api_key()
        analysis_mode = "lexicon"
        analysis_method = request.form.get("analysis_method", "hybrid")

        api_active = bool(api_key and is_api_available())

        if api_active and analysis_method == "full_ai":
            df = analyze_sentiment(df, text_column="clean_text", api_key=api_key)
            analysis_mode = "deepseek_ai"
        elif api_active and analysis_method == "hybrid":
            total_rows = len(df)
            if total_rows <= 200:
                df = analyze_sentiment(df, text_column="clean_text", api_key=api_key)
                analysis_mode = "deepseek_ai"
            else:
                # 1. Run lexicon analysis first for all data
                df = analyze_sentiment(df, text_column="clean_text", api_key=None)
                
                # 2. Sample 200 rows for DeepSeek AI
                sample_indices = df.sample(n=200, random_state=42).index
                sample_df = df.loc[sample_indices].copy()
                
                # 3. Analyze sample using DeepSeek
                sample_df = analyze_sentiment(sample_df, text_column="clean_text", api_key=api_key)
                
                # 4. Overwrite results back
                cols_to_update = ["sentiment", "emotion", "confidence", "positive_score", "negative_score", "neutral_score"]
                if "ai_reason" in sample_df.columns:
                    df["ai_reason"] = "Lexicon fallback"
                    cols_to_update.append("ai_reason")
                    
                df.loc[sample_indices, cols_to_update] = sample_df[cols_to_update]
                analysis_mode = "deepseek_ai_hybrid"
        else:
            df = analyze_sentiment(df, text_column="clean_text", api_key=None)
            analysis_mode = "lexicon"

        # Pastikan emotion lexicon terisi jika AI tidak memberikan emotion
        if "emotion" not in df.columns or df["emotion"].isna().all():
            from modules.emotion import detector
            emotions = df["clean_text"].apply(lambda x: detector.get_dominant_emotion(str(x))[0])
            df["emotion"] = emotions

        df["datetime_str"] = df["datetime"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
        )
        df["date_only"] = df["datetime"].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""
        )

        kpis = get_kpis(df)
        sentiment_index = get_sentiment_index(kpis)
        daily_df = aggregate_daily(df)
        account_df = aggregate_by_account(df)
        top_keywords = extract_top_keywords(df, text_column="clean_text", n=30)
        bigrams = extract_bigrams(df, text_column="clean_text", n=20)
        trigrams = extract_trigrams(df, text_column="clean_text", n=10)
        sentiment_spikes = detect_sentiment_spikes(daily_df, threshold_std=2.0)
        viral_posts = get_top_viral_posts(df, n=10)
        emotion_distribution = aggregate_by_emotion(df)
        weighted_sentiment = engagement_weighted_sentiment(df)

        summary_text = generate_executive_summary(
            kpis, daily_df, account_df, top_keywords, df,
            sentiment_spikes=sentiment_spikes,
            viral_posts=viral_posts,
            emotion_distribution=emotion_distribution,
            bigrams=bigrams,
            trigrams=trigrams,
        )

        csv_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{session_id}_result.csv")
        df.to_csv(csv_path, index=False)

        session["session_id"] = session_id
        session["dup_count"] = dup_count

        daily_json = json.loads(daily_df.to_json(orient="records", date_format="iso"))
        account_json = json.loads(account_df.to_json(orient="records"))
        detail_json = json.loads(df.to_json(orient="records", date_format="iso", default_handler=str))

        result = {
            "session_id": session_id,
            "kpis": kpis,
            "sentiment_index": sentiment_index,
            "daily": daily_json,
            "accounts": account_json,
            "keywords": [{"word": w, "count": c} for w, c in top_keywords],
            "bigrams": [{"phrase": w, "count": c} for w, c in bigrams],
            "trigrams": [{"phrase": w, "count": c} for w, c in trigrams],
            "sentiment_spikes": sentiment_spikes,
            "viral_posts": viral_posts,
            "emotion_distribution": emotion_distribution,
            "weighted_sentiment": weighted_sentiment,
            "summary": summary_text,
            "detail": detail_json,
            "total_rows": len(df),
            "dup_count": dup_count,
            "columns": [c for c in df.columns if c not in ("clean_text",)],
            "analysis_mode": analysis_mode,
        }
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Process error: {e}", exc_info=True)
        return jsonify({"error": f"Terjadi kesalahan saat memproses file: {str(e)}"}), 500


@app.route("/download/<session_id>/<format>")
def download(session_id, format):
    csv_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{session_id}_result.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "Data tidak ditemukan."}), 404

    df = pd.read_csv(csv_path)
    if format == "csv":
        content = export_to_csv(df)
        return Response(
            content,
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment;filename=analisis_sentimen_{session_id[:8]}.csv"},
        )
    elif format == "json":
        content = export_to_json(df)
        return Response(
            content,
            mimetype="application/json",
            headers={"Content-Disposition": f"attachment;filename=analisis_sentimen_{session_id[:8]}.json"},
        )
    return jsonify({"error": "Format tidak didukung."}), 400


@app.route("/download-summary/<session_id>")
def download_summary(session_id):
    csv_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{session_id}_result.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "Data tidak ditemukan."}), 404

    df = pd.read_csv(csv_path)
    df = calculate_engagement(df)
    df = calculate_engagement_weighted(df)
    df = preprocess_dataframe(df, text_column="Konten")
    df = analyze_sentiment(df, text_column="clean_text")
    kpis = get_kpis(df)
    daily_df = aggregate_daily(df)
    account_df = aggregate_by_account(df)
    top_keywords = extract_top_keywords(df, text_column="clean_text", n=30)
    bigrams = extract_bigrams(df, text_column="clean_text", n=20)
    trigrams = extract_trigrams(df, text_column="clean_text", n=10)
    sentiment_spikes = detect_sentiment_spikes(daily_df, threshold_std=2.0)
    viral_posts = get_top_viral_posts(df, n=10)
    emotion_distribution = aggregate_by_emotion(df)

    summary_text = generate_executive_summary(
        kpis, daily_df, account_df, top_keywords, df,
        sentiment_spikes=sentiment_spikes,
        viral_posts=viral_posts,
        emotion_distribution=emotion_distribution,
        bigrams=bigrams,
        trigrams=trigrams,
    )

    return Response(
        summary_text,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment;filename=executive_summary_{session_id[:8]}.md"},
    )


@app.route("/sample")
def download_sample():
    sample = [
        {"Tanggal": "01/06/2026", "Waktu": "08:15:00", "X akun": "@user1",
         "Konten": "Produk ini sangat bagus dan berkualitas tinggi, recommended!",
         "Komentar": 12, "Repost": 5, "Likes": 120, "Views": "1 rb", "Link": "https://x.com/user1/1"},
        {"Tanggal": "01/06/2026", "Waktu": "09:30:00", "X akun": "@user2",
         "Konten": "Pelayanan lambat dan mengecewakan, tidak sesuai ekspektasi",
         "Komentar": 45, "Repost": 23, "Likes": 67, "Views": "2,5 rb", "Link": "https://x.com/user2/1"},
        {"Tanggal": "01/06/2026", "Waktu": "14:00:00", "X akun": "@user1",
         "Konten": "Harga cukup terjangkau untuk kualitas segini",
         "Komentar": 5, "Repost": 2, "Likes": 30, "Views": "500", "Link": "https://x.com/user1/2"},
        {"Tanggal": "02/06/2026", "Waktu": "10:00:00", "X akun": "@user3",
         "Konten": "Barang rusak saat sampai, chat CS lama merespon",
         "Komentar": 89, "Repost": 34, "Likes": 45, "Views": "3 rb", "Link": "https://x.com/user3/1"},
        {"Tanggal": "02/06/2026", "Waktu": "16:45:00", "X akun": "@user2",
         "Konten": "Update terbaru aplikasinya bagus banget, fiturnya lengkap",
         "Komentar": 23, "Repost": 15, "Likes": 200, "Views": "5 rb", "Link": "https://x.com/user2/2"},
        {"Tanggal": "03/06/2026", "Waktu": "07:30:00", "X akun": "@user4",
         "Konten": "Pengiriman cepat dan packing rapi, terima kasih",
         "Komentar": 8, "Repost": 3, "Likes": 89, "Views": "1,2 rb", "Link": "https://x.com/user4/1"},
        {"Tanggal": "03/06/2026", "Waktu": "20:00:00", "X akun": "@user3",
         "Konten": "Biasa aja, gak ada yang spesial, standar banget",
         "Komentar": 15, "Repost": 2, "Likes": 22, "Views": "800", "Link": "https://x.com/user3/2"},
        {"Tanggal": "04/06/2026", "Waktu": "11:00:00", "X akun": "@user5",
         "Konten": "Sangat kecewa dengan kualitas produk kali ini, mending beli di toko lain",
         "Komentar": 67, "Repost": 28, "Likes": 34, "Views": "4 rb", "Link": "https://x.com/user5/1"},
        {"Tanggal": "04/06/2026", "Waktu": "15:30:00", "X akun": "@user1",
         "Konten": "Event kemarin seru banget, terima kasih panitia!",
         "Komentar": 34, "Repost": 45, "Likes": 300, "Views": "10 rb", "Link": "https://x.com/user1/3"},
        {"Tanggal": "05/06/2026", "Waktu": "09:00:00", "X akun": "@user6",
         "Konten": "Tolong perbaiki sistem notifikasinya, banyak bug",
         "Komentar": 56, "Repost": 12, "Likes": 78, "Views": "2 rb", "Link": "https://x.com/user6/1"},
        {"Tanggal": "05/06/2026", "Waktu": "18:00:00", "X akun": "@user4",
         "Konten": "Diskon besar-besaran akhir pekan ini, jangan lewatkan!",
         "Komentar": 120, "Repost": 89, "Likes": 450, "Views": "15 rb", "Link": "https://x.com/user4/2"},
        {"Tanggal": "06/06/2026", "Waktu": "08:00:00", "X akun": "@user7",
         "Konten": "Aplikasi sering force close, tolong segera diperbaiki",
         "Komentar": 200, "Repost": 56, "Likes": 90, "Views": "8 rb", "Link": "https://x.com/user7/1"},
    ]
    import csv
    import io
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=sample[0].keys())
    writer.writeheader()
    writer.writerows(sample)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sample_data.csv"},
    )


@app.route("/api/key-status")
def api_key_status():
    available = is_api_available()
    return jsonify({
        "has_key": available,
        "mode": "deepseek_ai" if available else "lexicon",
        "message": "DeepSeek AI aktif" if available else "Mode lexicon (tanpa API key)",
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
