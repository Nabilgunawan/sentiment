"""Quick model setup: saves base IndoBERT with classification head directly.
These models have random classifier weights and NEED proper training on Colab/Kaggle.
This just enables the inference pipeline to load Transformer models for emotion+sarcasm."""
import os, sys, json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

import config
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def save_model(task, output_dir, labels):
    model_name = "indobenchmark/indobert-base-p1"
    print(f"Loading {model_name} for {task} with {len(labels)} labels...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)},
        label2id={l: i for i, l in enumerate(labels)},
    )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_mapping = {
        "label2id": {l: i for i, l in enumerate(labels)},
        "id2label": {str(i): l for i, l in enumerate(labels)},
        "labels": labels,
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    print(f"[OK] {task} model saved to {output_dir} ({labels})")

save_model("emotion", "models/emotion", config.EMOTION_LABELS)
save_model("sarcasm", "models/sarcasm", config.SARCASM_LABELS)
print("\nDone! Model files saved. Run verify_models.py to test loading.")
