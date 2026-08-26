# Eksperimen Klasifikasi Teks (4 Label) — IndoNLU P2 Skenario 2

Eksperimen ini mereplikasi skema dari repo referensi (folder `reference/`), tetapi
dengan dua perbedaan utama:

1. **Label berjumlah 4** (bukan 2), sesuai task IndoNLU P2 skenario 2:
   - `non_porno`
   - `percakapan_porno`
   - `penyebar_konten_porno`
   - `penjaja_seks`
2. **Dijalankan di server (via SSH)**, bukan Kaggle/Colab. Karena itu semua kode
   memakai argumen command-line (bukan input interaktif), dan plot disimpan ke
   file (bukan `plt.show()`).

## Struktur Folder

```
experiment/
├── config.py                       # konfigurasi terpusat (label, path, default model)
├── dataset/
│   ├── train.csv                   # data latih (kolom: text, labels)
│   └── test.csv                    # data uji   (kolom: text, labels)
├── get_example/
│   └── generate_examples.py        # TAHAP 0: menyiapkan contoh few-shot
├── llm_inference/
│   ├── main.py                     # TAHAP 1 entry point (inferensi)
│   ├── inference.py                # logika inferensi massal
│   ├── prompts.py                  # template prompt (zero-shot & few-shot)
│   ├── model_utils.py              # load model & generasi teks
│   └── io_utils.py                 # baca/tulis JSONL
├── post_processing/
│   └── evaluate.py                 # TAHAP 2: bersihkan jawaban + evaluasi
├── output/                         # hasil inferensi & evaluasi (diabaikan git)
└── requirements.txt
```

## Setup & Instalasi (di server)

```bash
# 1) Aktifkan venv yang sudah ada di server
source /path/ke/venv/bin/activate

# 2) Cek GPU & versi CUDA yang didukung driver
nvidia-smi

# 3) Install torch versi CUDA yang sesuai GPU.
#    - RTX 50-series (Blackwell, mis. RTX 5060 Ti): WAJIB cu128 atau cu130
#    - GPU lain: sesuaikan (cu121 / cu124, dst.)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 4) Baru install dependensi lainnya (torch sudah terpasang, jadi dilewati)
pip install -r requirements.txt

# 5) Verifikasi GPU terbaca (harus keluar True dan versi mengandung +cu128)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> PENTING: jangan langsung `pip install -r requirements.txt` tanpa langkah 3,
> karena PyPI akan menginstall `torch` versi CPU (tidak memakai GPU).

## Alur Eksperimen (3 Tahap)

### Tahap 0 — Menyiapkan contoh few-shot

Mengambil contoh berlabel dari `train.csv` untuk tiap kelas, lalu menyisipkannya ke
setiap baris `test.csv`. Hasilnya file JSONL `test_with_example.jsonl`.

```bash
python get_example/generate_examples.py \
    --train_file dataset/train.csv \
    --test_file dataset/test.csv \
    --num_examples 5 \
    --output_file dataset/test_with_example.jsonl
```

### Tahap 1 — Inferensi LLM

Menjalankan model LLM untuk mengklasifikasi setiap teks uji.

```bash
# Mode FEW-SHOT
python llm_inference/main.py \
    --input_file_path dataset/test_with_example.jsonl \
    --output_file_path output/inference_results.jsonl \
    --prompt_type few \
    --num_examples 5 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --device cuda

# Mode ZERO-SHOT (tidak butuh contoh; input cukup test.csv)
python llm_inference/main.py \
    --input_file_path dataset/test.csv \
    --output_file_path output/inference_results_zero.jsonl \
    --prompt_type zero \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --device cuda
```

> Argumen penting: `--device` (`cuda`/`cpu`), `--dtype` (`bfloat16`/`float16`/`float32`),
> `--max_new_tokens`, `--return_scores`, `--verbose`.

### Tahap 2 — Pembersihan & Evaluasi

Membersihkan jawaban mentah model menjadi label kanonik, lalu menghitung metrik.

```bash
python post_processing/evaluate.py \
    --input_file output/inference_results.jsonl \
    --output_jsonl output/processed_results.jsonl \
    --output_csv output/evaluation_report.csv
```

Output: JSONL bersih, CSV laporan, dan gambar `evaluation_report_confusion_matrix.png`.

## Catatan untuk Server (SSH)

- Lihat bagian "Setup & Instalasi" di atas untuk langkah install lengkap. Intinya:
  install torch versi CUDA dulu, jangan sampai terinstall versi CPU.
- Jika server tidak punya GPU, jalankan dengan `--device cpu`.
- Untuk GPU lama (T4/Turing), gunakan `--dtype float16` (bfloat16 hanya cepat di
  GPU Ampere ke atas).
- Plot confusion matrix disimpan sebagai file PNG karena server tidak punya display.
