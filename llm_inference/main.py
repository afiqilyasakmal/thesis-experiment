"""
main.py
=======
TITIK MASUK (entry point) untuk TAHAP 1 — Inferensi LLM.

File ini hanya bertugas: (1) membaca argumen dari command line, lalu (2) memanggil
fungsi run_bulk_inference() di inference.py. Logika inti inferensi ada di modul
terpisah agar file ini tetap ringkas dan mudah dipahami.

Cara menjalankan (dari root proyek = folder experiment/), contoh mode FEW-SHOT:
    python llm_inference/main.py \
        --input_file_path dataset/test_with_example.jsonl \
        --output_file_path output/inference_results.jsonl \
        --prompt_type few \
        --num_examples 5 \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --device cuda \
        --return_scores

Contoh mode ZERO-SHOT (input cukup dataset/test.csv yang sudah berisi teks & label):
    python llm_inference/main.py \
        --input_file_path dataset/test.csv \
        --output_file_path output/inference_results_zero.jsonl \
        --prompt_type zero \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --device cuda

CATATAN: file ini dirancang untuk dijalankan di SERVER (via SSH), BUKAN Kaggle/Colab.
Oleh karena itu tidak ada kode mount drive, device_map="auto", atau dependensi
khusus notebook.
"""

import argparse
import sys
from pathlib import Path

# Pastikan root proyek (folder experiment/) masuk ke sys.path, sehingga `import config`
# dan `from llm_inference.xxx import ...` selalu berhasil dari mana pun file dijalankan.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_NUM_EXAMPLES,
)
from llm_inference.inference import run_bulk_inference


def main():
    parser = argparse.ArgumentParser(
        description="Inferensi klasifikasi teks 4 label (IndoNLU P2 skenario 2) memakai LLM."
    )
    parser.add_argument("--input_file_path", type=str, required=True,
                        help="Path file input (.jsonl atau .csv berisi teks & label).")
    parser.add_argument("--output_file_path", type=str, required=True,
                        help="Path file output hasil inferensi (.jsonl).")
    parser.add_argument("--prompt_type", type=str, choices=["zero", "few"], required=True,
                        help="Mode prompt: 'zero' (zero-shot) atau 'few' (few-shot).")
    parser.add_argument("--num_examples", type=int, default=DEFAULT_NUM_EXAMPLES,
                        help="Jumlah contoh per label (hanya dipakai saat prompt_type='few').")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME,
                        help="Nama model HuggingFace atau path lokal.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                        help="Device: 'cuda', 'cuda:0', atau 'cpu'.")
    parser.add_argument("--hf_token", type=str, default="",
                        help="Token HuggingFace (opsional, untuk model gated/private).")
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE,
                        choices=["bfloat16", "float16", "float32"],
                        help="Tipe data bobot model.")
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Maksimum token baru yang dihasilkan model.")
    parser.add_argument("--return_scores", action="store_true",
                        help="Simpan juga skor (probabilitas) tiap subtoken hasil.")
    parser.add_argument("--verbose", action="store_true",
                        help="Tampilkan progres per baris.")

    args = parser.parse_args()

    run_bulk_inference(
        input_file_path=args.input_file_path,
        output_file_path=args.output_file_path,
        prompt_type=args.prompt_type,
        num_examples=args.num_examples,
        model_name=args.model_name,
        device=args.device,
        hf_token=args.hf_token,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        return_scores=args.return_scores,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
