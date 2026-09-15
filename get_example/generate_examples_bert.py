"""
generate_examples_bert.py
=========================
TAHAP 0 (varian BERT) — Menyiapkan contoh few-shot berbasis kemiripan semantik
(Sentence-BERT). Adaptasi dari reference/get_example/get_example_bert.py (Alvaro).

Perbedaan dengan generate_examples.py (varian random):
    - Contoh dipilih berdasarkan KEMIRIPAN SEMANTIK (bukan acak): untuk tiap baris
      data uji, diambil top-N contoh latih paling mirip untuk SETIAP label (4 kelas).
    - Fungsi similaritas bisa dipilih: cosine / dot / euclidean / manhattan.

Struktur output SAMA persis dengan generate_examples.py, sehingga drop-in compatible
dengan tahap inferensi (llm_inference) dan prompt few-shot yang sudah ada:
    {
        "idx": 0,
        "text": "isi teks uji",
        "label": "label sebenarnya dari teks uji",
        "examples": {
            "non_porno":               [["teks", "label"], ...],
            "percakapan_porno":        [["teks", "label"], ...],
            "penyebar_konten_porno":   [["teks", "label"], ...],
            "penjaja_seks":            [["teks", "label"], ...]
        }
    }

Cara menjalankan (dari root proyek = folder experiment/):
    python get_example/generate_examples_bert.py \
        --train_file dataset/train.csv \
        --test_file dataset/test.csv \
        --num_examples 5 \
        --sim_function cosine \
        --output_file dataset/test_with_example_bert_cosine.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Pastikan root proyek masuk sys.path agar `import config` berhasil.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, SimilarityFunction

from config import LABELS, TEST_FILE, TRAIN_FILE
from trace import trace_input, trace_output, trace_running, trace_start

# Pemetaan nama fungsi similaritas -> konstanta sentence-transformers.
SIM_FUNCTIONS = {
    "cosine": SimilarityFunction.COSINE,
    "dot": SimilarityFunction.DOT_PRODUCT,
    "euclidean": SimilarityFunction.EUCLIDEAN,
    "manhattan": SimilarityFunction.MANHATTAN,
}


def generate_examples_bert(
    train_file: str,
    test_file: str,
    num_examples: int,
    output_file: str,
    bert_model: str = "indobenchmark/indobert-large-p2",
    sim_function: str = "cosine",
    drop_duplicate_text: bool = True,
) -> None:
    """
    Menyisipkan contoh few-shot (per kelas) berbasis similaritas semantik Sentence-BERT.

    Argumen:
        train_file          : path CSV data latih (kolom: text, labels).
        test_file           : path CSV data uji (kolom: text, labels).
        num_examples        : jumlah contoh yang diambil per label (top-N paling mirip).
        output_file         : path file JSONL keluaran.
        bert_model          : nama model SentenceTransformer (default mengikuti Alvaro).
        sim_function        : 'cosine' | 'dot' | 'euclidean' | 'manhattan'.
        drop_duplicate_text : buang teks duplikat pada data latih per label.

    CATATAN PENTING (perbedaan dari kode Alvaro yang kami perbaiki):
        model.similarity() dengan similarity_fn_name SELALU mengembalikan nilai dengan
        konvensi "semakin TINGGI = semakin mirip":
          - cosine / dot        : skor lebih tinggi = lebih mirip (natural).
          - euclidean / manhattan : sentence-transformers me-NEGASI jarak
            (util.euclidean_sim / manhattan_sim mengembalikan -jarak), sehingga nilai
            lebih tinggi tetap = lebih mirip.
        Karena itu, untuk SEMUA fungsi similaritas kita cukup ambil top-N nilai TERBESAR.
        (Kode Alvaro mengurutkan menaik untuk euclidean/manhattan, yang justru memilih
        contoh paling TIDAK mirip — bertentangan dengan maksud skripsi.)
    """
    if sim_function not in SIM_FUNCTIONS:
        raise ValueError(
            f"sim_function {sim_function!r} tidak dikenal. Pilih salah satu dari "
            f"{sorted(SIM_FUNCTIONS)}."
        )

    # 1) Baca data latih & uji.
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # 2) Kelompokkan data latih per label (4 kelas), buang duplikat bila diminta.
    examples_per_label = {}
    for label in LABELS:
        subset = train_df[train_df["labels"] == label].reset_index(drop=True)
        if drop_duplicate_text:
            subset = subset.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
        examples_per_label[label] = subset

    # 3) Validasi kecukupan contoh per label.
    for label in LABELS:
        if len(examples_per_label[label]) < num_examples:
            raise ValueError(
                f"Label {label!r} hanya punya {len(examples_per_label[label])} contoh di "
                f"train, padahal num_examples={num_examples}. Kurangi num_examples."
            )

    # 4) Jejak langkah (trace): ringkasan input tahap ini.
    trace_start("generate_examples_bert", "generate_examples_bert")
    trace_input(
        f"train_file={train_file} ({len(train_df)} baris), "
        f"test_file={test_file} ({len(test_df)} baris), "
        f"num_examples={num_examples}, sim_function={sim_function}, "
        f"bert_model={bert_model}, drop_duplicate_text={drop_duplicate_text}"
    )
    trace_running()

    # 5) Muat model SentenceTransformer dengan fungsi similaritas terpilih.
    print(f"Memuat model SentenceTransformer: {bert_model} (sim_function={sim_function})")
    model = SentenceTransformer(bert_model, similarity_fn_name=SIM_FUNCTIONS[sim_function])

    # 6) Encode SEMUA teks sekali (batch), lalu hitung similaritas per label.
    test_texts = test_df["text"].tolist()
    print(f"Encoding {len(test_texts)} teks uji ...")
    test_embeddings = model.encode(test_texts, show_progress_bar=True)

    # similaritas[label] berbentuk (n_test, n_contoh_label).
    similarities = {}
    for label in LABELS:
        texts = examples_per_label[label]["text"].tolist()
        print(f"Encoding {len(texts)} contoh '{label}' ...")
        label_embeddings = model.encode(texts, show_progress_bar=True)
        similarities[label] = model.similarity(test_embeddings, label_embeddings).cpu().numpy()

    # 7) Untuk tiap baris uji, ambil top-N contoh paling mirip per label.
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    first_record = None
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (_, row) in enumerate(test_df.iterrows()):
            examples = {}
            for label in LABELS:
                sims = similarities[label][idx]
                # Ambil top-N nilai TERBESAR (paling mirip) dalam urutan menurun.
                top_indices = np.argsort(sims)[-num_examples:][::-1]
                examples[label] = [
                    [
                        examples_per_label[label]["text"].iloc[i],
                        examples_per_label[label]["labels"].iloc[i],
                    ]
                    for i in top_indices
                ]

            record = {
                "idx": idx,
                "text": row["text"],
                "label": row["labels"],
                "examples": examples,
            }
            if first_record is None:
                first_record = record
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 100 == 0:
                print(f"Proses {idx + 1}/{len(test_df)} instans")

    trace_output(
        f"{len(test_df)} record ditulis ke {output_file}; "
        f"contoh baris: {json.dumps(first_record, ensure_ascii=False)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Menyiapkan contoh few-shot per label berbasis similaritas Sentence-BERT."
    )
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE,
                        help="Path CSV data latih.")
    parser.add_argument("--test_file", type=str, default=TEST_FILE,
                        help="Path CSV data uji.")
    parser.add_argument("--num_examples", type=int, default=5,
                        help="Jumlah contoh per label (top-N paling mirip).")
    parser.add_argument("--output_file", type=str, default="dataset/test_with_example_bert.jsonl",
                        help="Path file JSONL keluaran.")
    parser.add_argument("--bert_model", type=str, default="indobenchmark/indobert-large-p2",
                        help="Nama model SentenceTransformer (atau path lokal).")
    parser.add_argument("--sim_function", type=str, default="cosine",
                        choices=sorted(SIM_FUNCTIONS),
                        help="Fungsi similaritas: cosine/dot/euclidean/manhattan.")
    parser.add_argument("--drop_duplicate_text", type=lambda s: s.lower() in ("1", "true", "yes"),
                        default=True,
                        help="Buang teks duplikat pada data latih per label (default: True).")
    args = parser.parse_args()

    generate_examples_bert(
        train_file=args.train_file,
        test_file=args.test_file,
        num_examples=args.num_examples,
        output_file=args.output_file,
        bert_model=args.bert_model,
        sim_function=args.sim_function,
        drop_duplicate_text=args.drop_duplicate_text,
    )
