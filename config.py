"""
config.py
---------
Konfigurasi terpusat untuk platform Analisis Sentimen Media Sosial Indonesia.
Mengelola pengaturan model, device, batch size, dan path.
"""
import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ── Base Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", str(BASE_DIR / "uploads"))
MODELS_DIR = BASE_DIR / "models"
MODELS_SENTIMENT_DIR = MODELS_DIR / "sentiment"
MODELS_EMOTION_DIR = MODELS_DIR / "emotion"
MODELS_SARCASM_DIR = MODELS_DIR / "sarcasm"

# Ensure directories exist
for d in [UPLOAD_FOLDER, MODELS_DIR, MODELS_SENTIMENT_DIR, MODELS_EMOTION_DIR, MODELS_SARCASM_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Device Configuration ──────────────────────────────────────────────────────
def get_device():
    """Auto-detect CUDA GPU atau fallback ke CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU terdeteksi: {device_name}")
            return "cuda"
        else:
            logger.info("CUDA tidak tersedia, menggunakan CPU")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch tidak terinstall, menggunakan CPU")
        return "cpu"

DEVICE = os.environ.get("DEVICE", "auto")  # "auto", "cuda", "cpu"

def resolve_device():
    """Resolve device setting."""
    if DEVICE == "auto":
        return get_device()
    return DEVICE

# ── Model Configuration ──────────────────────────────────────────────────────
# Pre-trained models (HuggingFace)
SENTIMENT_MODEL_NAME = os.environ.get(
    "SENTIMENT_MODEL_NAME",
    "mdhugol/indonesia-bert-sentiment-classification"
)
EMOTION_MODEL_NAME = os.environ.get(
    "EMOTION_MODEL_NAME",
    "indobenchmark/indobert-base-p1"
)
SARCASM_MODEL_NAME = os.environ.get(
    "SARCASM_MODEL_NAME",
    "indobenchmark/indobert-base-p1"
)

# Fine-tuned model paths (jika ada)
SENTIMENT_FINETUNED_PATH = os.environ.get(
    "SENTIMENT_FINETUNED_PATH",
    str(MODELS_SENTIMENT_DIR)
)
EMOTION_FINETUNED_PATH = os.environ.get(
    "EMOTION_FINETUNED_PATH",
    str(MODELS_EMOTION_DIR)
)
SARCASM_FINETUNED_PATH = os.environ.get(
    "SARCASM_FINETUNED_PATH",
    str(MODELS_SARCASM_DIR)
)

# ── Inference Configuration ───────────────────────────────────────────────────
USE_TRANSFORMERS = os.environ.get("USE_TRANSFORMERS", "true").lower() == "true"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "128"))
INFERENCE_THREADS = int(os.environ.get("INFERENCE_THREADS", "4"))

# ── Training Configuration ────────────────────────────────────────────────────
TRAINING_EPOCHS = int(os.environ.get("TRAINING_EPOCHS", "5"))
TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE", "16"))
TRAINING_LEARNING_RATE = float(os.environ.get("TRAINING_LEARNING_RATE", "2e-5"))
TRAINING_WARMUP_RATIO = float(os.environ.get("TRAINING_WARMUP_RATIO", "0.1"))
TRAINING_VALIDATION_SPLIT = float(os.environ.get("TRAINING_VALIDATION_SPLIT", "0.2"))
TRAINING_MAX_SEQ_LENGTH = int(os.environ.get("TRAINING_MAX_SEQ_LENGTH", "128"))

# ── Topic Modeling Configuration ──────────────────────────────────────────────
TOPIC_MODEL_EMBEDDING = os.environ.get(
    "TOPIC_MODEL_EMBEDDING",
    "indobenchmark/indobert-base-p1"
)
TOPIC_MIN_TOPIC_SIZE = int(os.environ.get("TOPIC_MIN_TOPIC_SIZE", "10"))
TOPIC_NR_TOPICS = os.environ.get("TOPIC_NR_TOPICS", "auto")  # "auto" or int
TOPIC_TOP_N_WORDS = int(os.environ.get("TOPIC_TOP_N_WORDS", "10"))

# ── DeepSeek API Configuration (Summary Only) ────────────────────────────────
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_URL = os.environ.get(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/chat/completions"
)
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ── Flask Configuration ───────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(32).hex())
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_MB", "50")) * 1024 * 1024
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.environ.get("FLASK_PORT", "5000"))

# ── Export Configuration ──────────────────────────────────────────────────────
REPORT_COMPANY_NAME = os.environ.get("REPORT_COMPANY_NAME", "Pahamdata.com")
REPORT_LOGO_PATH = os.environ.get("REPORT_LOGO_PATH", "")

# ── Sentiment Labels ─────────────────────────────────────────────────────────
SENTIMENT_LABELS = ["Positive", "Neutral", "Negative"]
EMOTION_LABELS = ["Marah", "Senang", "Sedih", "Takut", "Jijik", "Terkejut", "Netral"]
SARCASM_LABELS = ["Not Sarcasm", "Sarcasm"]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
