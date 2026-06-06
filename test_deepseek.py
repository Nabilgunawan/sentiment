"""Script uji koneksi DeepSeek API"""
import sys
import os
sys.path.insert(0, '.')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

key = os.environ.get('DEEPSEEK_API_KEY', '').strip()

if not key or not key.startswith('sk-') or len(key) < 15:
    print('[ERROR] API key tidak ditemukan atau format salah.')
    print('        Pastikan file .env berisi: DEEPSEEK_API_KEY=sk-...')
    sys.exit(1)

masked = key[:8] + '...' + key[-4:]
print(f'[OK] API key ditemukan: {masked}')
print('[OK] File .env terbaca dengan benar')
print()
print('Menguji koneksi ke DeepSeek API...')

from modules.deepseek_client import analyze_batch

test_texts = [
    'Produk ini sangat bagus dan memuaskan, saya sangat puas!',
    'Pelayanan sangat buruk dan mengecewakan, tidak akan beli lagi.',
    'Biasa aja sih, standar.',
]

result = analyze_batch(test_texts, key)
if result:
    print('[OK] DeepSeek API berhasil terhubung!')
    print()
    for i, r in enumerate(result):
        print(f'Teks {i+1}: "{test_texts[i][:50]}..."')
        print(f'  Sentimen : {r["sentiment"]}')
        print(f'  Emosi    : {r["emotion"]}')
        print(f'  Confidence: {r["confidence"]:.2f}')
        print(f'  Alasan   : {r["reason"]}')
        print()
    print('[SIAP] Aplikasi siap dijalankan dengan DeepSeek AI!')
else:
    print('[ERROR] Koneksi ke DeepSeek API gagal.')
    print('        Periksa API key dan koneksi internet Anda.')
    sys.exit(1)
