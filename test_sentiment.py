import sys; sys.path.insert(0,'.')
from dotenv import load_dotenv; load_dotenv()
from services.model_inference import predict_sentiment
tests = [('Pelayanan sangat buruk dan mengecewakan!', 'Negative'), ('Produk ini luar biasa bagus, sangat puas!', 'Positive'), ('Biasa aja sih, standar lah', 'Neutral')]
for text, expected in tests:
    r = predict_sentiment(text)
    ok = 'OK' if r['sentiment'] == expected else 'MISMATCH'
    print(f'[{ok}] "{text[:50]}" -> {r["sentiment"]} ({r["confidence"]:.4f})')
