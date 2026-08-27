"""
prompts.py
==========
Tempat menyusun TEMPLATE PROMPT untuk LLM.

Ada dua mode prompt (mengikuti skema folder reference/):
    1. ZERO-SHOT : model diminta mengklasifikasi tanpa diberi contoh apa pun.
    2. FEW-SHOT  : model diberi beberapa contoh berlabel sebelum mengerjakan input.

Nama variabel (list_inference_input, formatted_example) sengaja mengikuti konvensi
folder reference/ agar mudah dirujuk silang dengan kode aslinya. Perbedaannya:
reference/ memakai 2 label (pornografi / non_pornografi), sedangkan eksperimen ini
memakai EMPAT label (lihat config.LABELS).
"""

from config import LABELS


def build_zero_shot_prompt(list_inference_input: list) -> str:
    """
    Menyusun prompt ZERO-SHOT untuk satu teks input.

    Argumen:
        list_inference_input : list atribut per record; teks berada di index [1]
                               (mengikuti konvensi reference/).

    Pada zero-shot, model tidak diberi contoh; ia hanya mengandalkan instruksi
    tugas + daftar kategori beserta deskripsinya.
    """
    text = list_inference_input[1]
    return f"""Tugas: Diberikan sebuah teks pada input berikut, tentukan apakah teks tersebut mengandung unsur pornografi dengan kategori tertentu atau tidak. 

Instruksi:
Keluaran label jawaban hanya dapat berupa: 'percakapan_porno', 'penyebar_konten_porno', atau 'penjaja_seks', 'non_porno'
⁠'percakapan_porno': Teks berisi obrolan, cerita, atau bahasan seksual namun tidak menyebarkan konten porno maupun menawarkan layanan seks.
⁠'penyebar_konten_porno': Teks yang membagikan atau mempromosikan media (foto, audio, atau video) pornografi maupun tautan ke suatu halaman web konten pornografi.
⁠'penjaja_seks': Teks yang memberikan penawaran layanan seksual baik prostitusi langsung maupun layanan online seperti layanan video call sex.
⁠'non_porno': Teks sama sekali tidak membahas terkait pornografi, baik obrolan terkait seksual maupun layanan prostitusi.
Jangan memberikan penjelasan atas jawaban Anda.

Giliran Anda:
Input:
- Teks: '{text}'

Jawaban:
"""


def _build_formatted_example(examples: dict, num_examples: int) -> str:
    """
    Menyusun blok contoh few-shot (formatted_example) menjadi teks bernomor.

    Contoh disusun selang-seling antar label (label pertama dulu, lalu label kedua,
    dst) agar representasi tiap kelas merata — mirip skema penyelang-selingan yang
    dipakai folder reference/.
    """
    lines = []
    for i in range(num_examples):
        for label in LABELS:
            # examples[label] adalah list of [teks, label]; ambil contoh ke-i.
            if i < len(examples.get(label, [])):
                teks_contoh = examples[label][i][0]  # index [0] = teks (bukan label)
                lines.append(f"{len(lines) + 1}. Teks: {teks_contoh}. Jawaban: {label}")
    return "\n".join(lines)


def build_few_shot_prompt(
    list_inference_input: list, examples: dict, num_examples: int
) -> str:
    """
    Menyusun prompt FEW-SHOT untuk satu teks input.

    Argumen:
        list_inference_input : list atribut per record; teks berada di index [1].
        examples             : dict {label: [[teks, label], ...]} dari tahap get_example.
        num_examples         : jumlah contoh per label yang dipakai.
    """
    text = list_inference_input[1]
    formatted_example = _build_formatted_example(examples, num_examples)
    return f"""Tugas: Diberikan sebuah teks pada input berikut, tentukan apakah teks tersebut mengandung unsur pornografi dengan kategori tertentu atau tidak. 

Instruksi:
Keluaran label jawaban hanya dapat berupa: 'percakapan_porno', 'penyebar_konten_porno', atau 'penjaja_seks', 'non_porno'
⁠'percakapan_porno': Teks berisi obrolan, cerita, atau bahasan seksual namun tidak menyebarkan konten porno maupun menawarkan layanan seks.
⁠'penyebar_konten_porno': Teks yang membagikan atau mempromosikan media (foto, audio, atau video) pornografi maupun tautan ke suatu halaman web konten pornografi.
⁠'penjaja_seks': Teks yang memberikan penawaran layanan seksual baik prostitusi langsung maupun layanan online seperti layanan video call sex.
⁠'non_porno': Teks sama sekali tidak membahas terkait pornografi, baik obrolan terkait seksual maupun layanan prostitusi.
Jangan memberikan penjelasan atas jawaban Anda.

Contoh:
{formatted_example}

Giliran Anda:
Input:
- Teks: '{text}'

Jawaban:
"""
