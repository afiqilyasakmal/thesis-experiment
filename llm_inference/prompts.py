"""
prompts.py
==========
Tempat menyusun TEMPLATE PROMPT untuk LLM.

Ada dua mode prompt (mengikuti skema folder reference/):
    1. ZERO-SHOT : model diminta mengklasifikasi tanpa diberi contoh apa pun.
    2. FEW-SHOT  : model diberi beberapa contoh berlabel sebelum mengerjakan input.

Karena eksperimen ini memakai EMPAT label (bukan dua seperti reference/), template
di sini disesuaikan agar model memilih SALAH SATU dari empat kategori tersebut.
"""

from config import LABELS, LABEL_DESCRIPTIONS


def _format_label_list() -> str:
    """
    Menyusun daftar label beserta deskripsinya menjadi blok teks untuk prompt.

    Contoh hasil (disingkat):
        - non_porno_non_prostitusi (teks normal ...)
        - percakapan_porno (percakapan ...)
        ...
    """
    lines = [f"- {label} ({LABEL_DESCRIPTIONS[label]})" for label in LABELS]
    return "\n".join(lines)


def build_zero_shot_prompt(text: str) -> str:
    """
    Menyusun prompt ZERO-SHOT untuk satu teks input.

    Pada zero-shot, model tidak diberi contoh; ia hanya mengandalkan instruksi
    tugas + daftar kategori.
    """
    label_list = _format_label_list()
    return f"""Tugas: Diberikan sebuah teks pada input berikut, tentukan kategori teks tersebut.

Kategori yang tersedia:
{label_list}

Instruksi:
- Keluarkan label jawaban hanya berupa SALAH SATU dari empat kategori di atas, persis seperti penulisannya.
- Jangan memberikan penjelasan atau kalimat tambahan apa pun.

Giliran Anda:
Input:
- Teks: '{text}'

Jawaban:
"""


def build_few_shot_prompt(text: str, examples: dict, num_examples: int) -> str:
    """
    Menyusun prompt FEW-SHOT untuk satu teks input.

    Argumen:
        text         : teks yang akan diklasifikasi.
        examples     : dict dengan struktur {label: [[teks, label], ...]}.
                       Dihasilkan oleh tahap get_example (generate_examples.py).
        num_examples : jumlah contoh per label yang dipakai.

    Cara kerja: contoh-contoh disusun selang-seling antar label (label pertama dulu,
    lalu label kedua, dst) agar representasi tiap kelas merata — mirip skema
    penyelang-selingan yang dipakai folder reference/.
    """
    label_list = _format_label_list()

    # Susun contoh secara selang-seling antar label.
    formatted_examples = []
    for i in range(num_examples):
        for label in LABELS:
            # examples[label] adalah list of [teks, label]; ambil contoh ke-i.
            if i < len(examples.get(label, [])):
                teks_contoh = examples[label][i][0]  # index [0] = teks (bukan label)
                formatted_examples.append(
                    f"{len(formatted_examples) + 1}. Teks: {teks_contoh}. Jawaban: {label}"
                )

    contoh_block = "\n".join(formatted_examples)

    return f"""Tugas: Diberikan sebuah teks pada input berikut, tentukan kategori teks tersebut.

Kategori yang tersedia:
{label_list}

Instruksi:
- Keluarkan label jawaban hanya berupa SALAH SATU dari empat kategori di atas, persis seperti penulisannya.
- Jangan memberikan penjelasan atau kalimat tambahan apa pun.

Contoh:
{contoh_block}

Giliran Anda:
Input:
- Teks: '{text}'

Jawaban:
"""
