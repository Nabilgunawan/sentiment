# Panduan Instalasi dan Deployment Model Transformer Indonesia (Phase 2)

Dokumen ini berisi panduan lengkap untuk melatih, mengekspor, mengunggah, dan melayani model-model berbasis IndoBERT Transformer (Analisis Sentimen, Klasifikasi Emosi, dan Deteksi Sarkasme) pada platform analisis sentimen ini.

---

## 1. Arsitektur Pipeline

```
TRAINING ENVIRONMENT (Cloud GPU)
Google Colab / Kaggle GPU (T4 / P100)
       ↓ (Fine-Tuning IndoBERT)
Model Export (.zip berisi config.json, pytorch_model.bin, tokenizer.json, label_mapping.json)
       ↓
DEPLOYMENT (Web App Server)
Upload Model via API (POST /train/<task>) atau Manual Extract ke /models/
       ↓ (Hot Reload - Zero Downtime)
Inference Service (Mixed Precision, GPU Auto-detection, CPU Fallback)
       ↓
Prediction API & Dashboard
```

---

## 2. Struktur Folder Modul Training & Model

Pembaruan Phase 2 menambahkan folder `/training` dan folder `/models` di root proyek:

```
sentiment-app/
├── models/                           # Direktori penyimpanan model aktif
│   ├── sentiment/                    # Model analisis sentimen IndoBERT
│   ├── emotion/                      # Model klasifikasi emosi IndoBERT
│   └── sarcasm/                      # Model deteksi sarkasme IndoBERT
│
└── training/                         # Modul training terpisah
    ├── data_utils.py                 # Parser dataset, validator, dan splitting
    ├── trainer.py                    # Training loops (Trainer API)
    ├── evaluator.py                  # Evaluator metrik & visualisasi
    ├── sentiment_train.py            # CLI Script training sentiment
    ├── emotion_train.py              # CLI Script training emotion
    ├── sarcasm_train.py              # CLI Script training sarcasm
    │
    ├── notebooks/                    # Jupyter Notebooks untuk Google Colab & Kaggle
    │   ├── train_sentiment.ipynb     # Colab Sentiment
    │   ├── train_emotion.ipynb       # Colab Emotion
    │   ├── train_sarcasm.ipynb       # Colab Sarcasm
    │   ├── kaggle_train_sentiment.ipynb # Kaggle Sentiment
    │   ├── kaggle_train_emotion.ipynb   # Kaggle Emotion
    │   └── kaggle_train_sarcasm.ipynb   # Kaggle Sarcasm
    │
    ├── datasets/                     # Tempat menyimpan file training data lokal
    ├── configs/                      # Konfigurasi hyperparameter
    └── outputs/                      # Hasil metrik evaluasi & visualisasi lokal
```

---

## 3. Panduan Training Model (Cloud GPU)

Proses training **TIDAK** dilakukan di web dashboard utama untuk menjaga stabilitas memori server. Pelatihan dilakukan di **Google Colab** atau **Kaggle** menggunakan akselerasi GPU.

### Opsi A: Google Colab
1. Buka Jupyter Notebook yang sesuai di `/training/notebooks/`:
   - [train_sentiment.ipynb](file:///c:/Users/Nabill/AppData/Local/Temp/opencode/sentiment-app/training/notebooks/train_sentiment.ipynb)
   - [train_emotion.ipynb](file:///c:/Users/Nabill/AppData/Local/Temp/opencode/sentiment-app/training/notebooks/train_emotion.ipynb)
   - [train_sarcasm.ipynb](file:///c:/Users/Nabill/AppData/Local/Temp/opencode/sentiment-app/training/notebooks/train_sarcasm.ipynb)
2. Unggah notebook tersebut ke Google Colab.
3. Ubah Runtime ke **GPU T4** atau GPU yang tersedia.
4. Jalankan sel secara berurutan. Notebook akan secara otomatis:
   - Menginstal library yang dibutuhkan (`transformers`, `datasets`, `accelerate`, dll.)
   - Menghubungkan Google Drive untuk menyimpan cadangan model.
   - Meminta Anda mengunggah berkas dataset (CSV atau XLSX).
   - Melakukan pembersihan data dan standardisasi label.
   - Menjalankan fine-tuning IndoBERT (5 epochs default) dengan early stopping.
   - Menguji keakuratan model, memproduksi classification report, dan menampilkan visualisasi Confusion Matrix.
   - Mengompres model hasil training menjadi berkas `.zip` dan menyalinnya ke Google Drive Anda.

### Opsi B: Kaggle Notebook
1. Buat notebook baru di Kaggle.
2. Aktifkan akselerator **GPU T4** atau **GPU P100** di menu kanan panel Kaggle.
3. Impor Jupyter Notebook versi Kaggle dari `/training/notebooks/`:
   - [kaggle_train_sentiment.ipynb](file:///c:/Users/Nabill/AppData/Local/Temp/opencode/sentiment-app/training/notebooks/kaggle_train_sentiment.ipynb)
   - [kaggle_train_emotion.ipynb](file:///c:/Users/Nabill/AppData/Local/Temp/opencode/sentiment-app/training/notebooks/kaggle_train_emotion.ipynb)
   - [kaggle_train_sarcasm.ipynb](file:///c:/Users/Nabill/AppData/Local/Temp/opencode/sentiment-app/training/notebooks/kaggle_train_sarcasm.ipynb)
4. Tambahkan dataset berlabel Anda ke input Kaggle.
5. Jalankan notebook untuk melatih model. Berkas model akhir `.zip` akan tersimpan di direktori output `/kaggle/working/` dan siap diunduh langsung ke lokal Anda.

---

## 4. Deployment Model ke Aplikasi Web

Setelah berkas ZIP model (`sentiment_model.zip`, `emotion_model.zip`, `sarcasm_model.zip`) berhasil diunduh dari Google Colab atau Kaggle, gunakan salah satu dari metode di bawah ini untuk deploy ke aplikasi.

### Opsi 1: Otomatis via API (Hot Reload - Tanpa Restart Aplikasi)
Anda dapat mengirimkan berkas model ZIP langsung ke server web menggunakan perintah `curl` atau script POST request. Ini akan secara otomatis menghapus model lama di folder `models/`, mengekstrak model baru, dan mereset status memori (hot reload) agar model langsung aktif:

```bash
# Upload model sentiment
curl -X POST -F "file=@sentiment_model.zip" http://localhost:5000/train/sentiment

# Upload model emotion
curl -X POST -F "file=@emotion_model.zip" http://localhost:5000/train/emotion

# Upload model sarcasm
curl -X POST -F "file=@sarcasm_model.zip" http://localhost:5000/train/sarcasm
```

**Response Sukses:**
```json
{
  "message": "Model sentiment berhasil diunggah, diekstrak, dan di-hot-reload.",
  "status": "success",
  "target_directory": "c:\\path\\to\\sentiment-app\\models\\sentiment"
}
```

### Opsi 2: Manual Copy & Paste
1. Hentikan aplikasi Flask jika sedang berjalan (opsional).
2. Ekstrak berkas model ZIP ke folder tujuannya masing-masing di dalam direktori proyek:
   - Isi `sentiment_model.zip` diekstrak ke: `/models/sentiment/`
   - Isi `emotion_model.zip` diekstrak ke: `/models/emotion/`
   - Isi `sarcasm_model.zip` diekstrak ke: `/models/sarcasm/`
3. Pastikan berkas penting seperti `config.json`, `pytorch_model.bin` (atau `model.safetensors`), `tokenizer.json`, dan `label_mapping.json` terletak langsung di dalam folder tersebut (bukan di dalam nested folder).
4. Nyalakan aplikasi. Model baru akan dimuat otomatis saat pertama kali inferensi dipanggil.

---

## 5. Dokumentasi API Endpoint Model

Layanan inferensi menyediakan endpoint JSON sebagai berikut:

### 1. Inferensi Prediksi
* **Endpoint:** `POST /predict`
* **Headers:** `Content-Type: application/json`
* **Request Body (Single):**
  ```json
  {
    "text": "Wah keren banget layanannya, tapi sayang internetnya sering mati."
  }
  ```
* **Response (Single):**
  ```json
  {
    "text": "Wah keren banget layanannya, tapi sayang internetnya sering mati.",
    "sentiment": {
      "sentiment": "Negative",
      "confidence": 0.9412,
      "prob_positive": 0.0212,
      "prob_neutral": 0.0376,
      "prob_negative": 0.9412
    },
    "emotion": {
      "emotion": "Sedih",
      "confidence": 0.8123,
      "scores": {
        "Marah": 0.1122,
        "Senang": 0.0211,
        "Sedih": 0.8123,
        "Takut": 0.0154,
        "Jijik": 0.0321,
        "Terkejut": 0.0034,
        "Netral": 0.0035
      }
    },
    "sarcasm": {
      "sarcasm": true,
      "confidence": 0.8872
    }
  }
  ```

* **Request Body (Batch):**
  ```json
  {
    "texts": [
      "Wah keren banget layanannya!",
      "Kecewa banget dengan produk ini."
    ]
  }
  ```
* **Response (Batch):**
  ```json
  {
    "total": 2,
    "predictions": [
      {
        "text": "Wah keren banget layanannya!",
        "sentiment": { ... },
        "emotion": { ... },
        "sarcasm": { ... }
      },
      {
        "text": "Kecewa banget dengan produk ini.",
        "sentiment": { ... },
        "emotion": { ... },
        "sarcasm": { ... }
      }
    ]
  }
  ```

### 2. Status Load Model & Hardware
* **Endpoint:** `GET /models/status`
* **Response:**
  ```json
  {
    "device": "cuda",
    "use_transformers": true,
    "sentiment": {
      "loaded": true,
      "mode": "Transformer",
      "model_name_or_path": "c:\\path\\to\\models\\sentiment"
    },
    "emotion": {
      "loaded": false,
      "mode": "Lexicon Fallback",
      "model_name_or_path": "indobenchmark/indobert-base-p1"
    },
    "sarcasm": {
      "loaded": true,
      "mode": "Transformer",
      "model_name_or_path": "c:\\path\\to\\models\\sarcasm"
    }
  }
  ```

### 3. Mengambil Metrik Evaluasi Model
* **Endpoint:** `GET /models/metrics`
* **Response:**
  Mengembalikan isi `metrics.json` dari model yang di-train jika tersedia (akurasi, F1, precision, recall, training history).

---

## 6. Optimisasi Performa & Fallback

Aplikasi memiliki optimisasi bawaan untuk runtime production:
1. **Dynamic Batching:** Membagi data berukuran besar menjadi batch-batch berukuran kecil (`config.BATCH_SIZE = 32` default) untuk menghindari kehabisan memori GPU (Out of Memory).
2. **Mixed Precision:** Menggunakan `torch.cuda.amp.autocast()` secara dinamis ketika GPU NVIDIA terdeteksi, mempercepat inferensi hingga 2x dengan kualitas prediksi yang sama.
3. **Lazy Loading:** Model hanya akan di-load ke RAM/VRAM saat pertama kali diperlukan (e.g. unggahan file pertama atau request prediksi pertama), menghemat memori server saat startup.
4. **Lexicon/Rule Fallback:** Jika file model Transformer tidak ditemukan, atau terjadi error loading, atau `USE_TRANSFORMERS = False` disetel pada `.env`, sistem akan otomatis melakukan fallback ke analisis berbasis kamus (Lexicon & regex rules). Ini menjamin fungsionalitas aplikasi tetap berjalan 100% tanpa error crash.
