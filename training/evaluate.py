"""
training/evaluate.py
--------------------
Script evaluasi umum untuk memuat model (sentimen, emosi, atau sarkasme) yang sudah dilatih
dan menguji performanya pada dataset test/evaluasi eksternal.
Menghasilkan metrik dan grafik evaluasi (confusion matrix, ROC curve).
"""

import argparse
import base64
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from training.data_utils import load_training_data
from training.evaluator import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained Sequence Classification Model")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["sentiment", "emotion", "sarcasm"],
        help="Tipe model yang dievaluasi (sentiment, emotion, sarcasm)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path ke dataset evaluasi CSV/XLSX"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        default="text",
        help="Nama kolom teks"
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Nama kolom label"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Override path direktori model (default dari config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "training" / "outputs"),
        help="Direktori hasil export grafik dan metrik"
    )

    args = parser.parse_args()

    # Determine model directory
    model_dir = args.model_dir
    if not model_dir:
        model_dir = {
            "sentiment": config.SENTIMENT_FINETUNED_PATH,
            "emotion": config.EMOTION_FINETUNED_PATH,
            "sarcasm": config.SARCASM_FINETUNED_PATH
        }[args.model_type]

    logger.info(f"Mulai evaluasi model '{args.model_type}' dari {model_dir}...")
    
    # 1. Load Data
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset {args.dataset} tidak ditemukan!")
        sys.exit(1)

    try:
        texts, labels = load_training_data(args.dataset, args.text_col, args.label_col)
    except Exception as e:
        logger.error(f"Gagal memuat data: {e}")
        sys.exit(1)

    labels = [str(l).strip() for l in labels]

    # Normalize labels based on type
    if args.model_type == "sentiment":
        label_map = {
            "0": "Negative", "Negatif": "Negative", "negative": "Negative",
            "1": "Neutral", "Netral": "Neutral", "neutral": "Neutral",
            "2": "Positive", "Positif": "Positive", "positive": "Positive"
        }
        labels = [label_map.get(l, l) for l in labels]
    elif args.model_type == "sarcasm":
        label_map = {
            "1": "Sarcasm", "Sarcasm": "Sarcasm", "sarcasm": "Sarcasm", "true": "Sarcasm", "True": "Sarcasm",
            "0": "Not Sarcasm", "Not Sarcasm": "Not Sarcasm", "not sarcasm": "Not Sarcasm", "false": "Not Sarcasm", "False": "Not Sarcasm"
        }
        labels = [label_map.get(l, l) for l in labels]

    # 2. Instantiate Evaluator
    try:
        evaluator = ModelEvaluator(args.model_type)
        if args.model_dir:
            evaluator.save_dir = Path(args.model_dir)
            evaluator._load_model()
    except Exception as e:
        logger.error(f"Gagal memuat model: {e}")
        sys.exit(1)

    # 3. Run full evaluation
    logger.info(f"Mengevaluasi {len(texts)} sampel...")
    try:
        report = evaluator.full_report(texts, labels)
    except Exception as e:
        logger.error(f"Gagal menjalankan evaluasi: {e}")
        sys.exit(1)

    # Save charts as files
    os.makedirs(args.output_dir, exist_ok=True)
    
    charts_paths = {}
    if "charts" in report:
        for chart_name, chart_bytes in report["charts"].items():
            filepath = os.path.join(args.output_dir, f"{args.model_type}_{chart_name}.png")
            with open(filepath, "wb") as f:
                f.write(chart_bytes)
            charts_paths[chart_name] = filepath
            logger.info(f"Grafik disimpan ke {filepath}")

    # Remove raw bytes for JSON output
    clean_report = report.copy()
    if "charts" in clean_report:
        clean_report["charts"] = charts_paths

    # Save metrik JSON
    metrics_file = os.path.join(args.output_dir, f"{args.model_type}_eval_report.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(clean_report, f, ensure_ascii=False, indent=2)

    logger.info(f"Metrik laporan evaluasi disimpan ke {metrics_file}")
    logger.info("=" * 60)
    logger.info(f"Hasil Evaluasi {args.model_type.upper()}:")
    logger.info(f"Accuracy: {clean_report['accuracy']:.4f}")
    logger.info(f"F1 Macro: {clean_report['f1']['macro']:.4f}")
    logger.info("Classification Report:")
    logger.info("\n" + clean_report["classification_report"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
