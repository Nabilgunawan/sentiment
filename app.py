import os
import uuid
import json
import pandas as pd
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify,
    send_file, session, redirect, url_for, Response,
)
from werkzeug.utils import secure_filename
from modules.parser import allowed_file, process_upload
from modules.preprocessor import preprocess_dataframe
from modules.sentiment import analyze_sentiment
from modules.aggregator import (
    calculate_engagement, aggregate_daily, aggregate_by_account,
    extract_top_keywords, get_kpis,
)
from modules.summary import generate_executive_summary
from modules.exporter import export_to_csv, export_to_json

app = Flask(__name__)
app.secret_key = os.urandom(32).hex()
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
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
        df = preprocess_dataframe(df, text_column="Konten", do_stemming=do_stemming)
        df = analyze_sentiment(df, text_column="clean_text")

        df["datetime_str"] = df["datetime"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
        )
        df["date_only"] = df["datetime"].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""
        )

        kpis = get_kpis(df)
        daily_df = aggregate_daily(df)
        account_df = aggregate_by_account(df)
        top_keywords = extract_top_keywords(df, text_column="clean_text", n=30)
        summary_text = generate_executive_summary(
            kpis, daily_df, account_df, top_keywords, df
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
            "daily": daily_json,
            "accounts": account_json,
            "keywords": [{"word": w, "count": c} for w, c in top_keywords],
            "summary": summary_text,
            "detail": detail_json,
            "total_rows": len(df),
            "dup_count": dup_count,
            "columns": [c for c in df.columns if c not in ("clean_text",)],
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
    df = preprocess_dataframe(df, text_column="Konten")
    df = analyze_sentiment(df, text_column="clean_text")
    kpis = get_kpis(df)
    daily_df = aggregate_daily(df)
    account_df = aggregate_by_account(df)
    top_keywords = extract_top_keywords(df, text_column="clean_text", n=30)
    summary_text = generate_executive_summary(kpis, daily_df, account_df, top_keywords, df)

    return Response(
        summary_text,
        mimetype="text/plain",
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


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
