"""
api/export.py
--------------
Blueprint untuk export laporan: DOCX, XLSX, CSV, JSON, Summary.
"""
import os
import io
import json
import logging
import pandas as pd
from flask import Blueprint, jsonify, Response, send_file, current_app

logger = logging.getLogger(__name__)

export_bp = Blueprint("export", __name__)


def _load_result_df(session_id: str):
    """Load result DataFrame dari CSV file."""
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    csv_path = os.path.join(upload_folder, f"{session_id}_result.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


@export_bp.route("/download/<session_id>/csv")
def download_csv(session_id):
    """Download hasil analisis sebagai CSV."""
    df = _load_result_df(session_id)
    if df is None:
        return jsonify({"error": "Data tidak ditemukan."}), 404

    try:
        from reporting.csv_export import generate_csv_bytes
        csv_bytes = generate_csv_bytes(df)
        return Response(
            csv_bytes,
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": f"attachment;filename=analisis_sentimen_{session_id[:8]}.csv"
            },
        )
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        # Fallback sederhana
        output = io.StringIO()
        df.to_csv(output, index=False)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment;filename=analisis_sentimen_{session_id[:8]}.csv"
            },
        )


@export_bp.route("/download/<session_id>/json")
def download_json(session_id):
    """Download hasil analisis sebagai JSON."""
    df = _load_result_df(session_id)
    if df is None:
        return jsonify({"error": "Data tidak ditemukan."}), 404

    try:
        from reporting.csv_export import export_to_json
        content = export_to_json(df)
        return Response(
            content,
            mimetype="application/json",
            headers={
                "Content-Disposition": f"attachment;filename=analisis_sentimen_{session_id[:8]}.json"
            },
        )
    except Exception as e:
        logger.error(f"JSON export error: {e}")
        content = df.to_json(orient="records", date_format="iso", default_handler=str)
        return Response(
            content,
            mimetype="application/json",
            headers={
                "Content-Disposition": f"attachment;filename=analisis_sentimen_{session_id[:8]}.json"
            },
        )


@export_bp.route("/download/<session_id>/xlsx")
def download_xlsx(session_id):
    """Download laporan Excel multi-sheet."""
    df = _load_result_df(session_id)
    if df is None:
        return jsonify({"error": "Data tidak ditemukan."}), 404

    try:
        from reporting.xlsx_report import generate_xlsx_report
        from analytics.aggregator import (
            calculate_engagement, calculate_engagement_weighted,
            aggregate_daily, aggregate_by_account, aggregate_by_emotion,
            aggregate_by_sarcasm, extract_top_keywords, get_kpis,
            get_sentiment_index, engagement_weighted_sentiment,
        )
        from analytics.viral_detector import get_top_viral_posts
        from analytics.influencer_analyzer import InfluencerAnalyzer
        from services.summary_service import generate_executive_summary

        # Recalculate aggregations from saved data
        kpis = get_kpis(df)
        daily_df = aggregate_daily(df)
        account_df = aggregate_by_account(df)
        emotion_dist = aggregate_by_emotion(df)
        sarcasm_dist = aggregate_by_sarcasm(df)
        viral_posts = get_top_viral_posts(df, n=10)

        influencer_analyzer = InfluencerAnalyzer()
        influencers = influencer_analyzer.get_top_influencers(df, n=10)

        data = {
            "df": df,
            "kpis": kpis,
            "sentiment_index": get_sentiment_index(kpis),
            "daily": daily_df,
            "accounts": account_df,
            "emotion_distribution": emotion_dist,
            "sarcasm_distribution": sarcasm_dist,
            "viral_posts": viral_posts,
            "influencers": influencers,
            "keywords": extract_top_keywords(df, text_column="clean_text", n=30),
            "summary_text": "",
            "session_id": session_id,
        }

        xlsx_bytes = generate_xlsx_report(data)
        return Response(
            xlsx_bytes,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment;filename=laporan_sentimen_{session_id[:8]}.xlsx"
            },
        )
    except Exception as e:
        logger.error(f"XLSX export error: {e}", exc_info=True)
        return jsonify({"error": f"Gagal membuat laporan Excel: {str(e)}"}), 500


@export_bp.route("/download/<session_id>/docx")
def download_docx(session_id):
    """Download laporan DOCX profesional multi-halaman."""
    df = _load_result_df(session_id)
    if df is None:
        return jsonify({"error": "Data tidak ditemukan."}), 404

    try:
        from reporting.docx_report import generate_docx_report
        from analytics.aggregator import (
            calculate_engagement, calculate_engagement_weighted,
            aggregate_daily, aggregate_weekly, aggregate_monthly,
            aggregate_by_account, aggregate_by_emotion, aggregate_by_sarcasm,
            extract_top_keywords, extract_bigrams, extract_trigrams,
            get_kpis, get_sentiment_index, engagement_weighted_sentiment,
        )
        from analytics.spike_detector import SpikeDetector
        from analytics.viral_detector import get_top_viral_posts
        from analytics.influencer_analyzer import InfluencerAnalyzer
        from services.summary_service import generate_executive_summary

        kpis = get_kpis(df)
        sentiment_index = get_sentiment_index(kpis)
        daily_df = aggregate_daily(df)
        weekly_df = aggregate_weekly(df)
        monthly_df = aggregate_monthly(df)
        account_df = aggregate_by_account(df)
        emotion_dist = aggregate_by_emotion(df)
        sarcasm_dist = aggregate_by_sarcasm(df)
        top_keywords = extract_top_keywords(df, text_column="clean_text", n=30)
        bigrams = extract_bigrams(df, text_column="clean_text", n=20)
        trigrams = extract_trigrams(df, text_column="clean_text", n=10)
        weighted_sentiment = engagement_weighted_sentiment(df)

        spike_detector = SpikeDetector()
        sentiment_spikes = spike_detector.detect_sentiment_spikes(daily_df)

        viral_posts = get_top_viral_posts(df, n=10)

        influencer_analyzer = InfluencerAnalyzer()
        influencers = influencer_analyzer.get_top_influencers(df, n=10)

        # Determine date range
        date_range = ""
        if "datetime" in df.columns:
            try:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                valid = df["datetime"].dropna()
                if len(valid) > 0:
                    date_range = f"{valid.min().strftime('%d %b %Y')} - {valid.max().strftime('%d %b %Y')}"
            except Exception:
                pass

        summary_text = generate_executive_summary(
            kpis, daily_df, account_df, top_keywords, df,
            sentiment_spikes=sentiment_spikes,
            viral_posts=viral_posts,
            emotion_distribution=emotion_dist,
            sarcasm_distribution=sarcasm_dist,
            bigrams=bigrams, trigrams=trigrams,
            weighted_sentiment=weighted_sentiment,
            sentiment_index=sentiment_index,
        )

        data = {
            "df": df,
            "kpis": kpis,
            "sentiment_index": sentiment_index,
            "daily": daily_df,
            "weekly": weekly_df,
            "monthly": monthly_df,
            "accounts": account_df,
            "emotion_distribution": emotion_dist,
            "sarcasm_distribution": sarcasm_dist,
            "sentiment_spikes": sentiment_spikes,
            "viral_posts": viral_posts,
            "influencers": influencers,
            "keywords": top_keywords,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "weighted_sentiment": weighted_sentiment,
            "topics": [],
            "summary_text": summary_text,
            "session_id": session_id,
            "analysis_mode": "IndoBERT Transformer",
            "date_range": date_range,
        }

        docx_bytes = generate_docx_report(data)
        return Response(
            docx_bytes,
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment;filename=laporan_sentimen_{session_id[:8]}.docx"
            },
        )
    except Exception as e:
        logger.error(f"DOCX export error: {e}", exc_info=True)
        return jsonify({"error": f"Gagal membuat laporan DOCX: {str(e)}"}), 500


@export_bp.route("/download-summary/<session_id>")
def download_summary(session_id):
    """Download executive summary sebagai Markdown."""
    df = _load_result_df(session_id)
    if df is None:
        return jsonify({"error": "Data tidak ditemukan."}), 404

    try:
        from analytics.aggregator import (
            calculate_engagement, calculate_engagement_weighted,
            aggregate_daily, aggregate_by_account, aggregate_by_emotion,
            aggregate_by_sarcasm, extract_top_keywords, extract_bigrams,
            extract_trigrams, get_kpis, get_sentiment_index,
            engagement_weighted_sentiment,
        )
        from analytics.spike_detector import SpikeDetector
        from analytics.viral_detector import get_top_viral_posts
        from services.summary_service import generate_executive_summary

        kpis = get_kpis(df)
        sentiment_index = get_sentiment_index(kpis)
        daily_df = aggregate_daily(df)
        account_df = aggregate_by_account(df)
        top_keywords = extract_top_keywords(df, text_column="clean_text", n=30)
        bigrams = extract_bigrams(df, text_column="clean_text", n=20)
        trigrams = extract_trigrams(df, text_column="clean_text", n=10)
        emotion_dist = aggregate_by_emotion(df)
        sarcasm_dist = aggregate_by_sarcasm(df)

        spike_detector = SpikeDetector()
        sentiment_spikes = spike_detector.detect_sentiment_spikes(daily_df)
        viral_posts = get_top_viral_posts(df, n=10)

        summary = generate_executive_summary(
            kpis, daily_df, account_df, top_keywords, df,
            sentiment_spikes=sentiment_spikes,
            viral_posts=viral_posts,
            emotion_distribution=emotion_dist,
            sarcasm_distribution=sarcasm_dist,
            bigrams=bigrams, trigrams=trigrams,
            sentiment_index=sentiment_index,
        )

        return Response(
            summary,
            mimetype="text/markdown",
            headers={
                "Content-Disposition": f"attachment;filename=executive_summary_{session_id[:8]}.md"
            },
        )
    except Exception as e:
        logger.error(f"Summary export error: {e}")
        return jsonify({"error": f"Gagal membuat summary: {str(e)}"}), 500
