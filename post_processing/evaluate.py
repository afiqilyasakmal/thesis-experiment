"""
evaluate.py
===========
TAHAP 2 — Membersihkan jawaban mentah LLM dan menghitung metrik evaluasi.

Alur di dalam file ini:
    1. Membaca file JSONL hasil inferensi (berisi kolom "original_answer" & "label").
    2. Membersihkan "original_answer" menjadi salah satu dari 4 label kanonik
       (fungsi normalize_answer) — karena jawaban model sering tidak persis.
    3. Menghitung metrik: accuracy, precision, recall, F1, dan confusion matrix.
    4. Menyimpan hasil bersih ke JSONL dan laporan ke CSV (+ gambar confusion matrix).

PERBEDAAN dengan folder reference/:
    - reference/ memakai input() interaktif; file ini memakai argumen CLI sehingga
      bisa dijalankan otomatis di SERVER (SSH, non-interaktif).
    - Plot confusion matrix disimpan ke file PNG (bukan plt.show()) karena server
      biasanya tidak punya layar/display.

Cara menjalankan (dari root proyek = folder experiment/):
    python post_processing/evaluate.py \
        --input_file output/inference_results.jsonl \
        --output_jsonl output/processed_results.jsonl \
        --output_csv output/evaluation_report.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Pastikan root proyek masuk sys.path agar `import config` berhasil.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Matplotlib dalam mode "Agg" (non-interaktif) supaya tidak butuh display di server.
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import LABELS
from trace import trace_input, trace_output, trace_running, trace_start


# Penanda NEGASI: kata/frasa yang berarti "bukan porno / bukan prostitusi".
# Ini dicek PALING AWAL (setelah cocok persis) supaya jawaban seperti
# "bukan porno" tidak salah masuk ke kelas yang mengandung kata "porno".
_NEGATION_MARKERS = [
    "non_porno", "non porno", "nonporno",
    "bukan porno", "tidak porno",
    "non pornografi", "bukan pornografi", "tidak pornografi",
    "non prostitusi", "bukan prostitusi", "tidak prostitusi",
]

# Penanda untuk kelas "penyebar_konten_porno" (mencakup berbagai bentuk kata:
# penyebar, menyebarkan, penyebaran, sebar, dst).
_PENYEBAR_MARKERS = ["penyebar", "menyebar", "sebar"]

# Penanda untuk kelas "penjaja_seks".
_PENJAJA_MARKERS = ["penjaja", "jasa seks", "prostitusi", "pelacur", "lonte", "psk"]


def normalize_answer(raw_answer):
    """
    Petakan jawaban mentah LLM ke salah satu 4 label kanonik (lihat config.LABELS).

    Urutan pengecekan sengaja diatur dari yang paling spesifik agar tidak salah
    tangkap, karena beberapa label mengandung kata mirip (kata "porno" muncul di
    lebih dari satu label). Heuristik ini tidak sempurna — hasilnya tetap perlu
    diperiksa lewat file CSV laporan.

    Mengembalikan:
        label kanonik (str) bila berhasil dipetakan, atau None bila tidak bisa.
    """
    text = (raw_answer or "").strip().lower()

    # 1) Cocok persis dengan salah satu label -> langsung kembalikan.
    if text in LABELS:
        return text

    # 2) Negasi -> kelas "non_porno" (paling awal agar aman).
    if any(marker in text for marker in _NEGATION_MARKERS):
        return "non_porno"

    # 3) Penyebar konten porno.
    if any(marker in text for marker in _PENYEBAR_MARKERS):
        return "penyebar_konten_porno"

    # 4) Penjaja seks / prostitusi.
    if any(marker in text for marker in _PENJAJA_MARKERS):
        return "penjaja_seks"

    # 5) Percakapan porno, sekaligus fallback untuk teks yang hanya menyebut
    #    "porno"/"pornografi" tanpa kata spesifik (kelas mayoritas ber-porno).
    if "percakapan" in text or "porno" in text or "pornografi" in text:
        return "percakapan_porno"

    return None


def evaluate(
    input_file: str,
    output_jsonl: str,
    output_csv: str,
    prompt_type: str = "",
    model_name: str = "",
) -> None:
    """Menjalankan seluruh proses pembersihan + evaluasi."""
    # Judul dinamis untuk laporan & confusion matrix:
    #   "zero shot: Qwen/Qwen2.5-7B-Instruct" / "few shot: ..."
    if prompt_type and model_name:
        title = f"{prompt_type} shot: {model_name}"
    elif model_name:
        title = model_name
    else:
        title = "Confusion Matrix (4 label)"

    trace_start("evaluate", "evaluate")
    trace_input(f"input_file={input_file}, title={title!r}")
    trace_running()

    # ---------- 1) Baca file hasil inferensi & bersihkan ----------
    records = []
    unmatchable_count = 0  # jawaban yang tidak bisa dipetakan ke label apa pun
    missing_count = 0      # baris yang tidak punya kolom wajib
    first_record = True    # penanda untuk jejak contoh (hanya record pertama)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            # Wajib ada kolom original_answer (hasil model) dan label (kebenaran).
            if "original_answer" not in obj or "label" not in obj:
                missing_count += 1
                continue

            predicted = normalize_answer(obj.get("original_answer"))

            # Jejak contoh pembersihan jawaban (hanya record pertama).
            if first_record:
                first_record = False
                trace_start("evaluate", "normalize_answer")
                trace_input(obj.get("original_answer"))
                trace_running()
                trace_output(predicted)

            if predicted is None:
                unmatchable_count += 1
                obj["postprocessed_answer"] = None
            else:
                obj["postprocessed_answer"] = predicted

            records.append(obj)

    # ---------- 2) Simpan hasil bersih (JSONL + CSV) ----------
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for obj in records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "original_answer", "postprocessed_answer", "label"])
        for obj in records:
            writer.writerow([
                obj.get("text", ""),
                obj.get("original_answer", ""),
                obj.get("postprocessed_answer", "UNPROCESSABLE"),
                obj.get("label", ""),
            ])

    trace_output(
        f"hasil bersih → {output_jsonl}; laporan → {output_csv}; "
        f"{unmatchable_count} jawaban tak terpetakan, {missing_count} field hilang"
    )

    # ---------- 3) Evaluasi metrik ----------
    y_true, y_pred = [], []
    for obj in records:
        if obj.get("postprocessed_answer") is not None and obj.get("label") is not None:
            y_true.append(obj["label"])
            y_pred.append(obj["postprocessed_answer"])

    print(f"\n--- Ringkasan Pemrosesan ---")
    print(f"Total record diproses : {len(records)}")
    print(f"Tidak bisa dipetakan  : {unmatchable_count}")
    print(f"Field hilang          : {missing_count}")

    if not y_true:
        print("Tidak ada data yang bisa dievaluasi.")
        return

    # ---------- 4) Laporan klasifikasi & confusion matrix ----------
    print("\n" + "=" * 60)
    print(f"EVALUASI — {title}")
    print("=" * 60)
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()

    cm_path = os.path.splitext(output_csv)[0] + "_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix disimpan di: {cm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bersihkan jawaban LLM dan evaluasi metrik (4 label)."
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path file JSONL hasil inferensi.")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path file JSONL hasil bersih.")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path file CSV laporan evaluasi.")
    parser.add_argument("--prompt_type", type=str, choices=["zero", "few"], default="",
                        help="Mode prompt (zero/few) untuk judul laporan — opsional.")
    parser.add_argument("--model_name", type=str, default="",
                        help="Nama model untuk judul laporan — opsional.")
    args = parser.parse_args()

    evaluate(
        args.input_file,
        args.output_jsonl,
        args.output_csv,
        prompt_type=args.prompt_type,
        model_name=args.model_name,
    )
