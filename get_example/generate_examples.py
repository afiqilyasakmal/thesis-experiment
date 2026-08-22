"""
generate_examples.py
====================
TAHAP 0 — Menyiapkan contoh few-shot (few-shot examples).

Tujuan:
    Untuk setiap baris data uji (test.csv), ambil sejumlah contoh berlabel dari
    data latih (train.csv). Contoh-contoh ini nantinya disisipkan ke dalam prompt
    few-shot pada TAHAP 1 (inferensi).

PERBEDAAN dengan folder reference/:
    - reference/ memakai 2 label, sehingga contoh diambil dari 2 kelompok
      (porno & non-porno) dan disimpan sebagai dua kolom terpisah.
    - Eksperimen ini memakai 4 label, sehingga contoh diambil dari 4 kelas sekaligus
      dan disimpan dalam SATU dict bernama "examples" (lebih rapi & mudah dibaca).

Struktur output (satu baris per data uji):
    {
        "idx": 0,
        "text": "isi teks uji",
        "label": "label sebenarnya dari teks uji",
        "examples": {
            "non_porno_non_prostitusi": [["teks", "label"], ...],
            "percakapan_porno":        [["teks", "label"], ...],
            "penyebar_konten_porno":   [["teks", "label"], ...],
            "penjaja_seks":            [["teks", "label"], ...]
        }
    }

Cara menjalankan (dari root proyek = folder experiment/):
    python get_example/generate_examples.py \
        --train_file dataset/train.csv \
        --test_file dataset/test.csv \
        --num_examples 5 \
        --output_file dataset/test_with_example.jsonl
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Pastikan root proyek masuk sys.path agar `import config` berhasil.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config import LABELS, TEST_FILE, TRAIN_FILE


def generate_examples(
    train_file: str,
    test_file: str,
    num_examples: int,
    output_file: str,
    seed: int = None,
) -> None:
    """
    Menyisipkan contoh few-shot (per kelas) ke setiap baris data uji.

    Argumen:
        train_file   : path CSV data latih (kolom: text, labels).
        test_file    : path CSV data uji (kolom: text, labels).
        num_examples : jumlah contoh yang diambil per label.
        output_file  : path file JSONL keluaran.
        seed         : seed untuk random, agar hasil bisa direproduksi.
    """
    if seed is not None:
        random.seed(seed)

    # Baca data latih & data uji menjadi DataFrame.
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Kelompokkan data latih per label.
    # Struktur: {label: list of (teks, label)}
    examples_per_label = {}
    for label in LABELS:
        subset = train_df[train_df["labels"] == label]
        examples_per_label[label] = list(zip(subset["text"], subset["labels"]))

    # Validasi: pastikan tiap label memiliki cukup contoh untuk diambil.
    for label in LABELS:
        if len(examples_per_label[label]) < num_examples:
            raise ValueError(
                f"Label {label!r} hanya punya {len(examples_per_label[label])} contoh di "
                f"train, padahal num_examples={num_examples}. Kurangi num_examples."
            )

    # Pastikan folder tujuan ada.
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(test_df.iterrows()):
            # Ambil `num_examples` contoh ACAK untuk SETIAP label.
            examples = {}
            for label in LABELS:
                samples = random.sample(examples_per_label[label], num_examples)
                # Ubah tuple -> list agar valid saat di-dump menjadi JSON.
                examples[label] = [list(pair) for pair in samples]

            record = {
                "idx": idx,
                "text": row["text"],
                "label": row["labels"],
                "examples": examples,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Selesai. File contoh few-shot tersimpan di: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Menyiapkan contoh few-shot per label dari data latih."
    )
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE,
                        help="Path CSV data latih.")
    parser.add_argument("--test_file", type=str, default=TEST_FILE,
                        help="Path CSV data uji.")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Jumlah contoh per label.")
    parser.add_argument("--output_file", type=str, default="dataset/test_with_example.jsonl",
                        help="Path file JSONL keluaran.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed random agar hasil reproducible.")
    args = parser.parse_args()

    generate_examples(
        train_file=args.train_file,
        test_file=args.test_file,
        num_examples=args.num_examples,
        output_file=args.output_file,
        seed=args.seed,
    )
