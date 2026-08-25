"""
config.py
=========
Konfigurasi terpusat untuk seluruh eksperimen.

Tujuan file ini adalah menaruh SEMUA nilai tetap (label, path, pengaturan default
model) di SATU tempat, sehingga tidak tersebar sebagai "magic string/number" di
banyak file. Kalau nanti ada yang berubah (misal jumlah label, nama folder, atau
nama model), cukup ubah di sini satu kali.

Catatan penting terkait label:
    Eksperimen ini memakai SKENARIO 2 dari task IndoNLU P2, yaitu klasifikasi dengan
    EMPAT kelas (bukan dua kelas seperti di folder reference/). Keempat label
    tersebut adalah:
        1. non_porno_non_prostitusi  -> teks normal
        2. percakapan_porno          -> percakapan berunsur pornografi
        3. penyebar_konten_porno     -> penyebar konten pornografi
        4. penjaja_seks              -> penawaran jasa seksual / prostitusi
"""

# ---------------------------------------------------------------------------
# 1. Definisi label (4 kelas)
# ---------------------------------------------------------------------------
# Urutan list ini sengaja mengikuti urutan label pada scenario 2 IndoNLU.
# Urutan ini dipakai konsisten oleh label2id / id2label dan file evaluasi.
LABELS = [
    "non_porno_non_prostitusi",
    "percakapan_porno",
    "penyebar_konten_porno",
    "penjaja_seks",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Deskripsi singkat tiap label dalam bahasa Indonesia.
# Dipakai saat menyusun prompt, supaya model paham arti tiap kategori.
LABEL_DESCRIPTIONS = {
    "non_porno_non_prostitusi": "teks normal yang TIDAK mengandung pornografi maupun prostitusi",
    "percakapan_porno": "percakapan yang mengandung unsur pornografi",
    "penyebar_konten_porno": "teks yang menyebarkan atau membagikan konten pornografi",
    "penjaja_seks": "teks yang menawarkan jasa seksual (prostitusi)",
}

# ---------------------------------------------------------------------------
# 2. Path dataset & output (relatif terhadap root proyek = folder experiment/)
# ---------------------------------------------------------------------------
TRAIN_FILE = "dataset/train.csv"
TEST_FILE = "dataset/test.csv"
OUTPUT_DIR = "output"

# ---------------------------------------------------------------------------
# 3. Nilai default untuk proses inferensi
# ---------------------------------------------------------------------------
# Nama model HuggingFace (atau path lokal). Silakan ganti sesuai model yang dipakai.
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Device untuk menjalankan model. Karena eksperimen ini dijalankan di server fasilkom
# (via SSH), gunakan "cuda" bila server punya GPU; kalau tidak, set "cpu".
DEFAULT_DEVICE = "cuda"

# Tipe data model. bfloat16 hemat memori & cepat di GPU modern (Ampere ke atas).
# Untuk GPU lama (mis. T4/Turing) ganti "float16" atau "float32".
DEFAULT_DTYPE = "bfloat16"

# Maksimum token baru yang dihasilkan model. Jawaban label biasanya pendek (< 10 token).
DEFAULT_MAX_NEW_TOKENS = 30

# Jumlah contoh per label pada mode few-shot (lihat get_example/ & prompts.py).
DEFAULT_NUM_EXAMPLES = 5

# ---------------------------------------------------------------------------
# 4. Daftar model yang dicoba dalam eksperimen (repo ID HuggingFace)
# ---------------------------------------------------------------------------
# Semua model di bawah bertipe CausalLM (decoder-only) varian -Instruct/chat,
# sehingga cocok dengan load_model_tokenizer() dan apply_chat_template().
#
# Catatan sebelum menjalankan:
#   - meta-llama/Llama-3.1-8B-Instruct adalah model GATED: harus klik "Agree"
#     di halaman model HuggingFace dulu, lalu lempar token lewat --hf_token.
#   - Model 8B (Llama-3.1-8B, SahabatAI-8B) dalam bfloat16 ~16GB: muat di 1 GPU
#     16GB tapi ketat; kalau OOM, ubah load_model_tokenizer() ke device_map="auto"
#     agar memakai 2 GPU sekaligus.
MODEL_LIST = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/deepseek-llm-7b-chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct",
]
