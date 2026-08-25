"""
inference.py
============
Mengatur proses INFERENSI MASSAL (bulk): membaca file JSONL input, membangun prompt
untuk setiap baris data, menjalankan model, lalu menulis hasilnya ke file output.

File ini adalah "inti" dari TAHAP 1. Ia tidak dipanggil langsung oleh pengguna,
melainkan dipanggil oleh main.py (entry point). Pemisahan ini membuat alur lebih
mudah dibaca dan diuji.
"""

import time

from llm_inference.io_utils import read_input, write_jsonl
from llm_inference.model_utils import generate_answer, load_model_tokenizer
from llm_inference.prompts import build_few_shot_prompt, build_zero_shot_prompt
from trace import trace_input, trace_output, trace_running, trace_start


def _build_prompt(record: dict, prompt_type: str, num_examples: int) -> str:
    """
    Membangun prompt untuk satu record data berdasarkan tipe prompt.

    Argumen:
        record       : satu dict hasil parsing JSONL (berisi "text", "label", "examples").
        prompt_type  : "zero" (zero-shot) atau "few" (few-shot).
        num_examples : jumlah contoh per label (hanya dipakai saat few-shot).
    """
    text = record["text"]

    if prompt_type == "zero":
        return build_zero_shot_prompt(text)
    elif prompt_type == "few":
        # record["examples"] berisi contoh-contoh few-shot dari tahap get_example.
        return build_few_shot_prompt(text, record.get("examples", {}), num_examples)
    else:
        raise ValueError(f"prompt_type tidak dikenal: {prompt_type!r} (pilih 'zero' atau 'few')")


def run_bulk_inference(
    input_file_path: str,
    output_file_path: str,
    prompt_type: str,
    num_examples: int,
    model_name: str,
    device: str,
    hf_token: str,
    dtype: str,
    max_new_tokens: int,
    return_scores: bool,
    verbose: bool,
) -> None:
    """
    Menjalankan inferensi untuk seluruh baris file JSONL dan menyimpan hasilnya.

    Alur di dalam fungsi ini:
        1. Muat model + tokenizer (sekali saja, di luar loop, agar efisien).
        2. Baca seluruh data JSONL.
        3. Untuk setiap baris: bangun prompt -> generate jawaban -> simpan ke record.
        4. Tulis seluruh record (yang sudah ditambah kolom hasil) ke file output.
    """
    # 0) Jejak langkah (trace): ringkasan tahap inferensi.
    trace_start("inference", "run_bulk_inference")
    trace_input(
        f"input={input_file_path}, prompt_type={prompt_type}, "
        f"model={model_name}, device={device}, dtype={dtype}"
    )

    # 1) Muat model & tokenizer satu kali untuk seluruh data.
    if verbose:
        print(f"Memuat model dan tokenizer: {model_name} (device={device}, dtype={dtype})")
    model, tokenizer = load_model_tokenizer(model_name, hf_token, device, dtype)

    # 2) Baca data JSONL.
    records = read_input(input_file_path)
    total = len(records)
    print(f"Total data yang akan diproses: {total}")

    # 3) Loop inferensi per baris.
    start_time = time.time()
    for i, record in enumerate(records):
        prompt = _build_prompt(record, prompt_type, num_examples)

        # Jejak contoh prompt (hanya record pertama, sebagai representasi).
        if i == 0:
            prompt_func = (
                "build_zero_shot_prompt" if prompt_type == "zero"
                else "build_few_shot_prompt"
            )
            trace_start("prompts", prompt_func)
            trace_input(record["text"])
            trace_running()
            trace_output(prompt)

        try:
            result = generate_answer(
                prompt, tokenizer, model, device, max_new_tokens, return_scores
            )
            if return_scores:
                answer, subtokens, scores = result
                record["original_answer"] = answer
                record["list_subtoken"] = subtokens
                record["list_subtoken_score"] = scores
            else:
                record["original_answer"] = result
        except Exception as e:
            # Jangan hentikan seluruh proses hanya karena satu baris gagal
            # (misal input terlalu panjang / OOM). Tandai dan lanjutkan.
            if verbose:
                print(f"[GAGAL] baris ke-{i + 1}: {e}")
            record["original_answer"] = "failed_to_get_inference_result"

        # Jejak contoh jawaban model (hanya record pertama).
        if i == 0:
            trace_start("model_utils", "generate_answer")
            trace_input(
                f"prompt + chat_template, max_new_tokens={max_new_tokens}, greedy (do_sample=False)"
            )
            trace_running()
            trace_output(record["original_answer"])

        if verbose:
            elapsed = time.time() - start_time
            print(f"Memproses {i + 1}/{total} (berjalan {elapsed:.2f} detik)")

    # 4) Simpan hasil.
    write_jsonl(records, output_file_path)
    print(f"Inferensi selesai. Hasil tersimpan di: {output_file_path}")

    # 5) Cetak 10 hasil pertama sebagai contoh ke terminal.
    print("\n" + "=" * 64)
    print("TOP 10 HASIL INFERENSI (10 record pertama)")
    print("=" * 64)
    for r in records[:10]:
        teks = (r.get("text") or "").replace("\n", " ")[:70]
        print(
            f"[{r.get('idx', '?')}] label={r.get('label', '-')} "
            f"| jawaban_model={r.get('original_answer', '-')}"
        )
        print(f"     teks: {teks}")
