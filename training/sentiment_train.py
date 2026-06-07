"""
training/sentiment_train.py
----------------------------
Script pelatihan (fine-tuning) model analisis sentimen Indonesia berbasis Transformer.
Mendukung IndoBERT / IndoBERTweet / Indonesian RoBERTa.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from training.data_utils import load_training_data, split_data, validate_labels
from training.trainer import ModelTrainer
from training.evaluator import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "training" / "datasets" / "sentiment_dataset.csv"),
        help="Path ke dataset CSV/XLSX berlabel"
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
        "--model_name",
        type=str,
        default="indobenchmark/indobert-base-p1",
        help="Base model HuggingFace (e.g. indobenchmark/indobert-base-p1 atau w11wo/indonesian-roberta-base-sentiment-classifier)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Jumlah epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "sentiment_model"),
        help="Direktori hasil export model"
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        default=str(PROJECT_ROOT / "training" / "outputs" / "sentiment_metrics.json"),
        help="File hasil ekspor metrik evaluasi"
    )

    args = parser.parse_args()

    # 1. Load Data
    logger.info(f"Memuat data dari {args.dataset}...")
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset {args.dataset} tidak ditemukan!")
        sys.exit(1)

    try:
        texts, labels = load_training_data(args.dataset, args.text_col, args.label_col)
    except Exception as e:
        logger.error(f"Gagal memuat data: {e}")
        sys.exit(1)

    # 2. Validate Labels
    # Sentiment labels: Negatif (0), Netral (1), Positif (2)
    # Kita dukung label string atau integer
    labels = [str(l).strip() for l in labels]
    valid_labels = ["0", "1", "2", "Negatif", "Netral", "Positif", "negative", "neutral", "positive"]
    is_valid, invalid_labels = validate_labels(labels, valid_labels)
    if not is_valid:
        logger.error(f"Ditemukan label tidak valid: {invalid_labels}. Label yang diizinkan: {valid_labels}")
        sys.exit(1)

    # Normalize labels ke 0, 1, 2 atau string Positif, Netral, Negatif
    label_map = {
        "0": "Negative", "Negatif": "Negative", "negative": "Negative",
        "1": "Neutral", "Netral": "Neutral", "neutral": "Neutral",
        "2": "Positive", "Positif": "Positive", "positive": "Positive"
    }
    labels = [label_map.get(l, l) for l in labels]
    sentiment_labels = ["Positive", "Neutral", "Negative"]

    # 3. Split Data
    train_texts, val_texts, train_labels, val_labels = split_data(texts, labels, test_size=0.2)

    # Update config paths dynamically
    config.SENTIMENT_MODEL_NAME = args.model_name
    config.SENTIMENT_FINETUNED_PATH = args.output_dir
    config.TRAINING_EPOCHS = args.epochs
    config.TRAINING_BATCH_SIZE = args.batch_size
    config.TRAINING_LEARNING_RATE = args.lr
    config.TRAINING_MAX_SEQ_LENGTH = args.max_length

    # Ensure output dirs exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)

    # 4. Fine-Tuning
    logger.info("Memulai pelatihan model...")
    trainer = ModelTrainer("sentiment")
    trainer.save_dir = Path(args.output_dir) # override save dir
    
    # Run train
    history = trainer.train(
        train_texts, train_labels,
        val_texts, val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # 5. Evaluate Model
    logger.info("Memulai evaluasi model...")
    evaluator = ModelEvaluator("sentiment")
    evaluator.save_dir = Path(args.output_dir)
    evaluator._load_model() # Reload trained model weights
    
    report = evaluator.evaluate(val_texts, val_labels)

    # Save metrics.json
    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["precision"],
        "recall": report["recall"],
        "f1": report["f1"],
        "confusion_matrix": report["confusion_matrix"],
        "classification_report": report["classification_report"],
        "training_history": history["history"],
        "best_epoch": history["best_epoch"],
        "best_val_loss": history["best_val_loss"],
        "training_time_seconds": history["training_time_seconds"]
    }

    with open(args.metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logger.info(f"Model berhasil dilatih dan disimpan ke {args.output_dir}")
    logger.info(f"Metrik evaluasi disimpan ke {args.metrics_file}")
    logger.info(f"Akurasi Akhir: {metrics['accuracy']:.4f} | F1-Score Macro: {metrics['f1']['macro']:.4f}")


if __name__ == "__main__":
    main()
