import streamlit as st
import os
import re
from transformers import MBartTokenizerFast as MBartTokenizer, MBartForConditionalGeneration

# Mengatur konfigurasi halaman Streamlit agar terlihat premium dan menarik
st.set_page_config(
    page_title="Peringkas Berita Indonesia",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk meningkatkan estetika antarmuka (UI/UX)
st.markdown("""
    <style>
    /* Styling font dan background */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Judul Utama */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }
    
    /* Subjudul */
    .subtitle {
        font-size: 1.1rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Desain Container Box untuk Hasil */
    .result-container {
        background-color: #F8FAFC;
        border-left: 5px solid #FF4B4B;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: #94A3B8;
        text-align: center;
        padding: 10px;
        font-size: 0.85rem;
        border-top: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# Path ke folder model lokal
# Menggunakan pengecekan dinamis agar tetap berfungsi baik dijalankan dari root maupun folder nlp-summarizer
model_dir = "./best_model/"
if not os.path.exists(model_dir):
    # Fallback ke path absolut satu level di atas jika dijalankan dari dalam direktori nlp-summarizer
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "best_model"))

# Fungsi pembersihan output teks (Claude's Smart Trim + Emergency Clean)
def clean_output(text):
    # Menghapus spasi berlebih
    text = " ".join(text.split())
    
    # Memotong kalimat yang menggantung/tidak lengkap di akhir teks menggunakan regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if sentences and not sentences[-1].strip().endswith(('.', '!', '?')):
        sentences = sentences[:-1]
    final_text = " ".join(sentences) if sentences else text
    
    # Emergency clean: menghapus tanda baca di awal teks yang ditinggalkan oleh mBART
    final_text = final_text.lstrip(",.-_ ")
    
    # Memastikan huruf pertama menggunakan huruf kapital
    if final_text:
        final_text = final_text[0].upper() + final_text[1:]
        
    return final_text

# Cache pemuatan model dan tokenizer untuk menghemat memori dan meningkatkan performa aplikasi
@st.cache_resource
def load_model_and_tokenizer(path):
    # Memuat tokenizer menggunakan MBartTokenizer dari path lokal
    tokenizer = MBartTokenizer.from_pretrained(path)
    # Memuat model menggunakan MBartForConditionalGeneration dari path lokal
    model = MBartForConditionalGeneration.from_pretrained(path)
    return tokenizer, model

# Menampilkan indikator loading saat pertama kali memuat model
with st.spinner("Memuat model NLP... Harap tunggu sebentar."):
    try:
        tokenizer, model = load_model_and_tokenizer(model_dir)
    except Exception as e:
        st.error(f"Gagal memuat model dari '{model_dir}'. Detail error: {str(e)}")
        st.stop()

# Header Antarmuka Pengguna (UI)
st.markdown("<h1 class='main-title'>Peringkas Berita Indonesia</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ringkas artikel berita panjang Anda secara otomatis menggunakan model Transformer mBART yang telah di-fine-tune.</p>", unsafe_allow_html=True)

# Area Input Teks
text_input = st.text_area(
    label="Masukkan teks berita:",
    placeholder="Tempel atau ketik teks artikel berita Bahasa Indonesia di sini...",
    height=300
)

# Tombol Aksi
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    submit_button = st.button("Ringkas Sekarang", use_container_width=True)

# Logika ketika tombol ditekan
if submit_button:
    # Memeriksa jika input teks kosong
    if not text_input.strip():
        st.warning("Masukkan teks terlebih dahulu!")
    else:
        # Menampilkan spinner proses peringkasan
        with st.spinner("Sedang meringkas..."):
            try:
                # Tokenisasi teks input
                inputs = tokenizer(
                    text_input,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Menghasilkan ringkasan dengan parameter yang ditentukan
                summary_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    num_beams=4,
                    max_length=256,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                
                # Mendekode hasil token kembali menjadi teks biasa
                summary_text = tokenizer.decode(
                    summary_ids[0],
                    skip_special_tokens=True
                )
                
                # Melakukan pembersihan akhir pada teks hasil ringkasan
                cleaned_summary = clean_output(summary_text)
                
                # Menampilkan hasil ringkasan ke layar
                st.success("Hasil Ringkasan:")
                st.info(cleaned_summary)
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat meringkas teks: {str(e)}")

# Footer halaman aplikasi
st.markdown("""
    <div class="footer">
        Dibuat oleh <b>Muhammad Nazlul Ramadhyan</b> (2308107010036) | Proyek Akhir MK Pemrosesan Bahasa Alami
    </div>
""", unsafe_allow_html=True)
