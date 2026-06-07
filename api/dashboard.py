"""
api/dashboard.py
-----------------
Blueprint untuk endpoint dashboard tambahan.
"""
import os
import csv
import io
import logging
import pandas as pd
from flask import Blueprint, request, jsonify, Response

from services.deepseek_client import is_api_available

logger = logging.getLogger(__name__)

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/api/key-status")
def api_key_status():
    """Cek status API key (DeepSeek untuk summary)."""
    available = is_api_available()
    return jsonify({
        "has_key": available,
        "mode": "indobert_transformer",
        "summary_ai": "deepseek_active" if available else "template_fallback",
        "message": (
            "IndoBERT Transformer aktif | DeepSeek AI Summary aktif"
            if available
            else "IndoBERT Transformer aktif | Summary mode: Template"
        ),
    })


@dashboard_bp.route("/sample")
def download_sample():
    """Download sample CSV untuk testing."""
    sample = [
        {
            "Tanggal": "01/06/2026", "Waktu": "08:15:00", "X akun": "@user1",
            "Konten": "Produk ini sangat bagus dan berkualitas tinggi, recommended!",
            "Komentar": 12, "Repost": 5, "Likes": 120, "Views": "1 rb",
            "Link": "https://x.com/user1/1",
        },
        {
            "Tanggal": "01/06/2026", "Waktu": "09:30:00", "X akun": "@user2",
            "Konten": "Pelayanan lambat dan mengecewakan, tidak sesuai ekspektasi",
            "Komentar": 45, "Repost": 23, "Likes": 67, "Views": "2,5 rb",
            "Link": "https://x.com/user2/1",
        },
        {
            "Tanggal": "01/06/2026", "Waktu": "14:00:00", "X akun": "@user1",
            "Konten": "Harga cukup terjangkau untuk kualitas segini",
            "Komentar": 5, "Repost": 2, "Likes": 30, "Views": "500",
            "Link": "https://x.com/user1/2",
        },
        {
            "Tanggal": "02/06/2026", "Waktu": "10:00:00", "X akun": "@user3",
            "Konten": "Barang rusak saat sampai, chat CS lama merespon",
            "Komentar": 89, "Repost": 34, "Likes": 45, "Views": "3 rb",
            "Link": "https://x.com/user3/1",
        },
        {
            "Tanggal": "02/06/2026", "Waktu": "16:45:00", "X akun": "@user2",
            "Konten": "Update terbaru aplikasinya bagus banget, fiturnya lengkap",
            "Komentar": 23, "Repost": 15, "Likes": 200, "Views": "5 rb",
            "Link": "https://x.com/user2/2",
        },
        {
            "Tanggal": "03/06/2026", "Waktu": "07:30:00", "X akun": "@user4",
            "Konten": "Wah hebat sekali pemerintah, harga naik semua",
            "Komentar": 156, "Repost": 89, "Likes": 34, "Views": "8 rb",
            "Link": "https://x.com/user4/1",
        },
        {
            "Tanggal": "03/06/2026", "Waktu": "20:00:00", "X akun": "@user3",
            "Konten": "Biasa aja, gak ada yang spesial, standar banget",
            "Komentar": 15, "Repost": 2, "Likes": 22, "Views": "800",
            "Link": "https://x.com/user3/2",
        },
        {
            "Tanggal": "04/06/2026", "Waktu": "11:00:00", "X akun": "@user5",
            "Konten": "Sangat kecewa dengan kualitas produk kali ini, mending beli di toko lain",
            "Komentar": 67, "Repost": 28, "Likes": 34, "Views": "4 rb",
            "Link": "https://x.com/user5/1",
        },
        {
            "Tanggal": "04/06/2026", "Waktu": "15:30:00", "X akun": "@user1",
            "Konten": "Event kemarin seru banget, terima kasih panitia!",
            "Komentar": 34, "Repost": 45, "Likes": 300, "Views": "10 rb",
            "Link": "https://x.com/user1/3",
        },
        {
            "Tanggal": "05/06/2026", "Waktu": "09:00:00", "X akun": "@user6",
            "Konten": "Tolong perbaiki sistem notifikasinya, banyak bug",
            "Komentar": 56, "Repost": 12, "Likes": 78, "Views": "2 rb",
            "Link": "https://x.com/user6/1",
        },
        {
            "Tanggal": "05/06/2026", "Waktu": "18:00:00", "X akun": "@user4",
            "Konten": "Diskon besar-besaran akhir pekan ini, jangan lewatkan!",
            "Komentar": 120, "Repost": 89, "Likes": 450, "Views": "15 rb",
            "Link": "https://x.com/user4/2",
        },
        {
            "Tanggal": "06/06/2026", "Waktu": "08:00:00", "X akun": "@user7",
            "Konten": "Aplikasi sering force close, tolong segera diperbaiki",
            "Komentar": 200, "Repost": 56, "Likes": 90, "Views": "8 rb",
            "Link": "https://x.com/user7/1",
        },
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=sample[0].keys())
    writer.writeheader()
    writer.writerows(sample)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=sample_data.csv"},
    )
