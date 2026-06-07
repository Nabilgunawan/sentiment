"""
api/upload.py
--------------
Blueprint untuk upload file dan menjalankan pipeline analisis lengkap.

Pipeline:
1. Upload & validasi file
2. Parse file (CSV/XLSX/HTML)
3. Preprocessing (cleaning + normalisasi)
4. Sentiment Classification (IndoBERT)
5. Emotion Detection
6. Sarcasm Detection
7. Topic Extraction
8. Aggregation & Analytics
9. Executive Summary Generation
10. Return JSON response
"""
import os
import uuid
import json
import logging
import pandas as pd
from datetime import datetime

from flask import Blueprint, request, jsonify, session, current_app

logger = logging.getLogger(__name__)

upload_bp = Blueprint("upload", __name__)


def _lazy_load_services():
    """Lazy load semua services untuk menghindari circular import dan mempercepat startup."""
    from modules.parser import allowed_file, process_upload
    from preprocessing.pipeline import preprocess_dataframe
    import services.model_inference as model_inference
    from services.summary_service import generate_executive_summary
    from analytics.aggregator import (
        calculate_engagement, calculate_engagement_weighted,
        aggregate_daily, aggregate_weekly, aggregate_monthly,
        aggregate_by_account, aggregate_by_emotion, aggregate_by_sarcasm,
        aggregate_emotion_daily,
        extract_top_keywords, extract_bigrams, extract_trigrams,
        engagement_weighted_sentiment,
        get_kpis, get_sentiment_index,
    )
    from analytics.spike_detector import SpikeDetector
    from analytics.viral_detector import get_top_viral_posts
    from analytics.influencer_analyzer import InfluencerAnalyzer

    return {
        "allowed_file": allowed_file,
        "process_upload": process_upload,
        "preprocess_dataframe": preprocess_dataframe,
        "model_inference": model_inference,
        "generate_executive_summary": generate_executive_summary,
        "calculate_engagement": calculate_engagement,
        "calculate_engagement_weighted": calculate_engagement_weighted,
        "aggregate_daily": aggregate_daily,
        "aggregate_weekly": aggregate_weekly,
        "aggregate_monthly": aggregate_monthly,
        "aggregate_by_account": aggregate_by_account,
        "aggregate_by_emotion": aggregate_by_emotion,
        "aggregate_by_sarcasm": aggregate_by_sarcasm,
        "aggregate_emotion_daily": aggregate_emotion_daily,
        "extract_top_keywords": extract_top_keywords,
        "extract_bigrams": extract_bigrams,
        "extract_trigrams": extract_trigrams,
        "engagement_weighted_sentiment": engagement_weighted_sentiment,
        "get_kpis": get_kpis,
        "get_sentiment_index": get_sentiment_index,
        "SpikeDetector": SpikeDetector,
        "get_top_viral_posts": get_top_viral_posts,
        "InfluencerAnalyzer": InfluencerAnalyzer,
    }


# Singleton service instances (lazy loaded)
_services = {}


def _get_services():
    global _services
    if not _services:
        _services = _lazy_load_services()
    return _services


@upload_bp.route("/upload", methods=["POST"])
def upload_file():
    """Upload file dan jalankan pipeline analisis lengkap."""
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    svc = _get_services()

    if not svc["allowed_file"](file.filename):
        return jsonify({
            "error": "Format file tidak didukung. Gunakan: .csv, .xlsx, atau .html"
        }), 400

    # Generate session ID
    session_id = uuid.uuid4().hex
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{session_id}.{ext}"
    upload_folder = current_app.config["UPLOAD_FOLDER"]
    filepath = os.path.join(upload_folder, safe_name)
    file.save(filepath)

    try:
        # ── Parse Options ──
        opts = json.loads(request.form.get("options", "{}"))
        run_sentiment  = opts.get("sentiment",  True)
        run_emotion    = opts.get("emotion",    True)
        run_sarcasm    = opts.get("sarcasm",    True)
        run_topics     = opts.get("topics",     True)
        run_ai_summary = opts.get("ai_summary", True)
        logger.info(
            f"[{session_id[:8]}] Options: sentiment={run_sentiment}, emotion={run_emotion}, "
            f"sarcasm={run_sarcasm}, topics={run_topics}, ai_summary={run_ai_summary}"
        )

        # ── Step 1: Parse & Validate ──
        logger.info(f"[{session_id[:8]}] Step 1: Parsing file...")
        df, dup_count = svc["process_upload"](filepath, file.filename)

        # ── Step 2: Calculate Engagement ──
        logger.info(f"[{session_id[:8]}] Step 2: Calculating engagement...")
        df = svc["calculate_engagement"](df)
        df = svc["calculate_engagement_weighted"](df)

        # ── Step 3: Preprocessing ──
        logger.info(f"[{session_id[:8]}] Step 3: Preprocessing...")
        df = svc["preprocess_dataframe"](df, text_column="Konten")

        texts = df["clean_text"].fillna("").tolist()

        # ── Step 4: Sentiment Classification (IndoBERT) ──
        if run_sentiment:
            logger.info(f"[{session_id[:8]}] Step 4: Sentiment analysis (IndoBERT)...")
            sentiment_results = svc["model_inference"].batch_predict(texts, task="sentiment")
            df["sentiment"]    = [r["sentiment"]    for r in sentiment_results]
            df["confidence"]   = [r["confidence"]   for r in sentiment_results]
            df["prob_positive"] = [r["prob_positive"] for r in sentiment_results]
            df["prob_neutral"]  = [r["prob_neutral"]  for r in sentiment_results]
            df["prob_negative"] = [r["prob_negative"] for r in sentiment_results]
        else:
            logger.info(f"[{session_id[:8]}] Step 4: Sentiment SKIPPED")
            df["sentiment"]    = "Neutral"
            df["confidence"]   = 0.0
            df["prob_positive"] = 0.0
            df["prob_neutral"]  = 1.0
            df["prob_negative"] = 0.0

        # ── Step 5: Emotion Detection ──
        if run_emotion:
            logger.info(f"[{session_id[:8]}] Step 5: Emotion detection...")
            emotion_results = svc["model_inference"].batch_predict(texts, task="emotion")
            df["emotion"]            = [r["emotion"]    for r in emotion_results]
            df["emotion_confidence"] = [r["confidence"] for r in emotion_results]
        else:
            logger.info(f"[{session_id[:8]}] Step 5: Emotion SKIPPED")
            df["emotion"]            = "Netral"
            df["emotion_confidence"] = 0.0

        # ── Step 6: Sarcasm Detection ──
        if run_sarcasm:
            logger.info(f"[{session_id[:8]}] Step 6: Sarcasm detection...")
            sarcasm_results = svc["model_inference"].batch_predict(texts, task="sarcasm")
            df["sarcasm"]            = [r["sarcasm"]    for r in sarcasm_results]
            df["sarcasm_confidence"] = [r["confidence"] for r in sarcasm_results]
        else:
            logger.info(f"[{session_id[:8]}] Step 6: Sarcasm SKIPPED")
            df["sarcasm"]            = False
            df["sarcasm_confidence"] = 0.0

        # ── Step 7: Topic Extraction ──
        topics_data = {"topics": [], "topic_over_time": []}
        if run_topics:
            logger.info(f"[{session_id[:8]}] Step 7: Topic extraction...")
            try:
                from services.topic_service import TopicService
                topic_svc = TopicService()
                topics_data = topic_svc.extract_topics(texts)
            except Exception as e:
                logger.warning(f"Topic extraction gagal: {e}")
        else:
            logger.info(f"[{session_id[:8]}] Step 7: Topics SKIPPED")

        # ── Step 8: Aggregation & Analytics ──
        logger.info(f"[{session_id[:8]}] Step 8: Aggregation...")

        # Datetime formatting
        df["datetime_str"] = df["datetime"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(x) else ""
        )
        df["date_only"] = df["datetime"].apply(
            lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""
        )

        kpis = svc["get_kpis"](df)
        sentiment_index = svc["get_sentiment_index"](kpis)
        daily_df = svc["aggregate_daily"](df)
        weekly_df = svc["aggregate_weekly"](df)
        monthly_df = svc["aggregate_monthly"](df)
        account_df = svc["aggregate_by_account"](df)
        top_keywords = svc["extract_top_keywords"](df, text_column="clean_text", n=30)
        bigrams = svc["extract_bigrams"](df, text_column="clean_text", n=20)
        trigrams = svc["extract_trigrams"](df, text_column="clean_text", n=10)

        emotion_distribution = svc["aggregate_by_emotion"](df)    if run_emotion  else {}
        sarcasm_distribution = svc["aggregate_by_sarcasm"](df)    if run_sarcasm  else {}
        emotion_daily        = svc["aggregate_emotion_daily"](df)  if run_emotion  else pd.DataFrame()
        weighted_sentiment   = svc["engagement_weighted_sentiment"](df)

        spike_detector = svc["SpikeDetector"]()
        sentiment_spikes = spike_detector.detect_sentiment_spikes(daily_df)

        viral_posts = svc["get_top_viral_posts"](df, n=10)

        influencer_analyzer = svc["InfluencerAnalyzer"]()
        influencers = influencer_analyzer.get_top_influencers(df, n=10)

        # ── Step 9: Executive Summary ──
        logger.info(f"[{session_id[:8]}] Step 9: Executive summary...")
        summary_text = svc["generate_executive_summary"](
            kpis, daily_df, account_df, top_keywords, df,
            sentiment_spikes=sentiment_spikes,
            viral_posts=viral_posts,
            emotion_distribution=emotion_distribution,
            sarcasm_distribution=sarcasm_distribution,
            topics=topics_data.get("topics", []),
            bigrams=bigrams,
            trigrams=trigrams,
            weighted_sentiment=weighted_sentiment,
            sentiment_index=sentiment_index,
            use_ai=run_ai_summary,
        )

        # ── Step 10: Save & Return ──
        logger.info(f"[{session_id[:8]}] Step 10: Saving results...")
        csv_path = os.path.join(upload_folder, f"{session_id}_result.csv")
        df.to_csv(csv_path, index=False)

        session["session_id"] = session_id
        session["dup_count"] = dup_count

        # Serialize DataFrames
        daily_json = json.loads(daily_df.to_json(orient="records", date_format="iso"))
        weekly_json = json.loads(weekly_df.to_json(orient="records", date_format="iso"))
        monthly_json = json.loads(monthly_df.to_json(orient="records", date_format="iso"))
        account_json = json.loads(account_df.to_json(orient="records"))
        emotion_daily_json = (
            json.loads(emotion_daily.to_json(orient="records", date_format="iso"))
            if not emotion_daily.empty else []
        )
        detail_json = json.loads(
            df.to_json(orient="records", date_format="iso", default_handler=str)
        )

        result = {
            "session_id": session_id,
            "kpis": kpis,
            "sentiment_index": sentiment_index,
            "daily": daily_json,
            "weekly": weekly_json,
            "monthly": monthly_json,
            "accounts": account_json,
            "keywords": [{"word": w, "count": c} for w, c in top_keywords],
            "bigrams": [{"phrase": w, "count": c} for w, c in bigrams],
            "trigrams": [{"phrase": w, "count": c} for w, c in trigrams],
            "sentiment_spikes": sentiment_spikes,
            "viral_posts": viral_posts,
            "emotion_distribution": emotion_distribution,
            "sarcasm_distribution": sarcasm_distribution,
            "emotion_daily": emotion_daily_json,
            "weighted_sentiment": weighted_sentiment,
            "topics": topics_data.get("topics", []),
            "topic_over_time": topics_data.get("topic_over_time", []),
            "influencers": influencers,
            "summary": summary_text,
            "detail": detail_json,
            "total_rows": len(df),
            "dup_count": dup_count,
            "columns": [c for c in df.columns if c not in ("clean_text",)],
            "analysis_mode": "indobert_transformer",
            "ran_options": {
                "sentiment":  run_sentiment,
                "emotion":    run_emotion,
                "sarcasm":    run_sarcasm,
                "topics":     run_topics,
                "ai_summary": run_ai_summary,
            },
        }

        logger.info(
            f"[{session_id[:8]}] Analisis selesai: {len(df)} post, "
            f"{kpis['positive']} positif, {kpis['neutral']} netral, {kpis['negative']} negatif"
        )
        return jsonify(result)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Process error: {e}", exc_info=True)
        return jsonify({"error": f"Terjadi kesalahan saat memproses file: {str(e)}"}), 500
