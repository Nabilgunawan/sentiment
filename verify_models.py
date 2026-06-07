"""Script verifikasi download & loading model IndoBERT"""
import os, sys, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

import config
from services.model_inference import load_sentiment_model, load_emotion_model, load_sarcasm_model

print(f"Device         : {config.resolve_device()}")
print(f"USE_TRANSFORMERS: {config.USE_TRANSFORMERS}")
print(f"Sentiment model : {config.SENTIMENT_MODEL_NAME}")
print(f"Emotion model   : {config.EMOTION_MODEL_NAME}")
print(f"Sarcasm model   : {config.SARCASM_MODEL_NAME}")
print(f"Finetuned paths : sentiment={config.SENTIMENT_FINETUNED_PATH}, emotion={config.EMOTION_FINETUNED_PATH}, sarcasm={config.SARCASM_FINETUNED_PATH}")
print()

for task, loader in [("sentiment", load_sentiment_model), ("emotion", load_emotion_model), ("sarcasm", load_sarcasm_model)]:
    print(f"--- Loading {task} model ---")
    model, tokenizer = loader()
    if model is not None:
        print(f"[OK] {task} TRANSFORMER loaded successfully")
    else:
        print(f"[FALLBACK] {task} menggunakan lexicon/rule-based (model Transformer tidak dimuat)")
    print()
