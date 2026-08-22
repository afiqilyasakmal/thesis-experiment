"""
io_utils.py
===========
Kumpulan fungsi utilitas untuk membaca dan menulis file JSONL (.jsonl).

Format JSONL: setiap baris berisi SATU objek JSON (dict). Format ini dipakai sebagai
"jembatan" antar-tahap eksperimen, sehingga setiap tahap bisa dijalankan terpisah
dan hasilnya bisa diperiksa manual:

    Tahap 0 (get_example)   -> menghasilkan file .jsonl berisi contoh few-shot
    Tahap 1 (llm_inference) -> membaca .jsonl, menambah kolom hasil, menulis .jsonl
    Tahap 2 (post_processing)-> membaca .jsonl, menambah kolom bersih, evaluasi
"""

import json
import os


def read_jsonl(path: str) -> list:
    """
    Membaca seluruh baris file JSONL menjadi list of dict.

    Argumen:
        path : path menuju file .jsonl

    Mengembalikan:
        list of dict, satu dict untuk setiap baris file.
    """
    result = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # lewati baris kosong agar tidak memicu error json.loads
                result.append(json.loads(line))
    return result


def write_jsonl(data: list, path: str) -> None:
    """
    Menulis list of dict ke file JSONL (satu dict = satu baris).

    Argumen:
        data : list of dict
        path : path tujuan file .jsonl (folder akan dibuat otomatis bila belum ada)
    """
    # Pastikan folder tujuan ada sebelum menulis
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in data:
            # ensure_ascii=False agar karakter non-ASCII (huruf Indonesia) tidak ter-escape
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_jsonl_keys(path: str) -> list:
    """
    Mengembalikan daftar nama kolom (key) dari baris pertama file JSONL.

    Berguna untuk memeriksa struktur data sebelum diproses (misal memastikan
    kolom "text", "label", "examples" sudah ada).
    """
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        return list(json.loads(first_line).keys())
