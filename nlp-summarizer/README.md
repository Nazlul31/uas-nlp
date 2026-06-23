# Peringkas Berita Indonesia

Aplikasi web berbasis **Streamlit** untuk meringkas berita berbahasa Indonesia secara otomatis menggunakan pendekatan **Abstractive Summarization**. Aplikasi ini menggunakan model *Transformer* **mBART** (`indobenchmark/indobart`) yang telah melalui proses *fine-tuning* dan disimpan secara lokal.

Proyek ini dibangun sebagai demonstrasi langsung (live demo) untuk Proyek Akhir Mata Kuliah **Pemrosesan Bahasa Alami**.

---

## 🚀 Fitur Utama
- **Ringkasan Abstrak Otomatis**: Menghasilkan ringkasan berita yang alami dan ringkas (bukan sekadar memotong kalimat asli).
- **Pembersihan Cerdas (Smart Post-processing)**: Dilengkapi dengan algoritma pembersih otomatis untuk mendeteksi kalimat menggantung di akhir paragraf, tanda baca liar dari model mBART, serta perbaikan otomatis huruf kapital di awal kalimat.
- **Resource Caching**: Menggunakan fitur `@st.cache_resource` agar model berukuran besar hanya dimuat sekali di memori saat aplikasi pertama kali dijalankan, menghemat penggunaan RAM dan mempercepat waktu tunggu pengguna.

---

## 📁 Struktur Proyek
```text
nlp-summarizer/
├── best_model/         # Folder lokal penyimpan bobot model & tokenizer (model.safetensors, dll)
├── app.py              # Logika utama aplikasi Streamlit dan UI/UX kustom
└── requirements.txt    # Daftar dependensi pustaka Python
```

---

## 🛠️ Cara Instalasi & Menjalankan Aplikasi

### 1. Prasyarat
Pastikan komputer Anda sudah terpasang **Python 3.8 ke atas**.

### 2. Kloning / Masuk ke Folder Proyek
Buka terminal Anda (Command Prompt/PowerShell/Bash) lalu navigasikan ke folder proyek:
```bash
cd nlp-summarizer
```

### 3. Unduh Folder Model Lokal (`best_model`)
Karena berkas model berukuran besar (~526MB), folder model tidak diunggah ke repositori git. Anda wajib mengunduhnya terlebih dahulu:
1. Unduh isi folder `best_model` dari tautan Google Drive berikut: [Link Google Drive](https://drive.google.com/drive/folders/1MRGfqazU7xe9aZVOvo9rdij7T9D7XNqa?usp=sharing)
2. Buat folder baru bernama `best_model` di dalam direktori `nlp-summarizer/` jika belum ada.
3. Ekstrak/pindahkan semua berkas hasil unduhan (`model.safetensors`, `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`) langsung ke dalam folder `best_model/` tersebut.

### 4. Instalasi Dependensi Pustaka
Instal pustaka Python yang didefinisikan di dalam `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Catatan: Instalasi ini mencakup Streamlit, PyTorch, Hugging Face Transformers, SentencePiece, dan Accelerate.*

### 5. Menjalankan Aplikasi
Jalankan server aplikasi lokal Streamlit dengan perintah:
```bash
streamlit run app.py
```

Setelah berhasil dijalankan, Streamlit akan otomatis membuka browser default Anda dan mengarahkan ke halaman aplikasi (biasanya di `http://localhost:8501`).

---

## ⚙️ Spesifikasi Model Peringkas
- **Arsitektur**: `MBartForConditionalGeneration`
- **Model Dasar**: `indobenchmark/indobart`
- **Tokenizer**: `MBartTokenizerFast` (memuat konfigurasi dari `tokenizer.json` lokal)
- **Konfigurasi Generasi Teks**:
  - `num_beams = 4` (Pencarian balok untuk kualitas teks optimal)
  - `max_length = 256` (Panjang token maksimal hasil ringkasan)
  - `early_stopping = True` (Berhenti otomatis setelah kalimat lengkap terbentuk)
  - `no_repeat_ngram_size = 3` (Mencegah pengulangan frasa/kata berulang)

---

## 👥 Identitas Mahasiswa
- **Nama**: Muhammad Nazlul Ramadhyan
- **NPM**: 2308107010036


