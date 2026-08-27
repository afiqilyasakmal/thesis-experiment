"""
model_utils.py
==============
Fungsi untuk memuat model & tokenizer, serta menjalankan inferensi (generasi teks).

PERBEDAAN PENTING dengan folder reference/:
    - Kode reference/ didesain untuk Kaggle/Colab, sehingga memakai
      `device_map="auto"` dan `torch_dtype=bfloat16` secara hard-coded.
    - Kode ini didesain untuk dijalankan di SERVER (via SSH). Oleh karena itu,
      pemilihan device ("cuda", "cuda:0", "cpu") dan tipe data (dtype) dibuat
      EKSPLISIT dan dapat diatur lewat argumen CLI (lihat main.py).
      Pengecualian: bila terdeteksi >1 GPU CUDA, model dimuat dengan
      device_map="auto" supaya bisa di-shard ke beberapa GPU (untuk model 8B).
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace import trace_input, trace_output, trace_running, trace_start


def load_model_tokenizer(
    model_name: str,
    hf_token: str = "",
    device: str = "cuda",
    dtype: str = "bfloat16",
):
    """
    Memuat model kausal (CausalLM) dan tokenizer-nya dari HuggingFace / path lokal.

    Argumen:
        model_name : nama model HuggingFace (mis. "Qwen/Qwen2.5-7B-Instruct")
                     atau path lokal menuju checkpoint model.
        hf_token   : token HuggingFace (opsional, hanya untuk model gated/private).
        device     : device tempat model diletakkan, mis. "cuda", "cuda:0", "cpu".
        dtype      : tipe data bobot model, salah satu "bfloat16" / "float16" / "float32".

    Mengembalikan:
        tuple (model, tokenizer).
    """
    trace_start("model_utils", "load_model_tokenizer")
    trace_input(f"model_name={model_name}, device={device}, dtype={dtype}")
    trace_running()

    # Jika disediakan token, login dulu agar bisa mengakses model gated.
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token)

    # Petakan string dtype menjadi tipe torch yang sesuai.
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype, torch.bfloat16)

    # Deteksi ketersediaan & jumlah GPU lebih dulu, karena menentukan cara memuat model.
    cuda_available = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if cuda_available else 0

    # Jatuh ke CPU bila GPU (CUDA) tidak tersedia di mesin.
    if device.startswith("cuda") and not cuda_available:
        print("[PERINGATAN] CUDA tidak tersedia di mesin ini, beralih ke CPU.")
        device = "cpu"

    # Sharding 1 model ke >1 GPU: pakai device_map="auto" supaya bobot dibelah otomatis
    # (mis. Llama-8B / SahabatAI-8B ~16GB -> 8GB di GPU 0 + 8GB di GPU 1). Tanpa ini,
    # model 8B akan OOM di satu GPU 8GB. Kalau cuma 1 GPU, tetap pakai .to(device) supaya
    # device sepenuhnya kita kontrol lewat argumen (lebih eksplisit & mudah di-debug).
    use_device_map_auto = device.startswith("cuda") and num_gpus > 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token or None)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        token=hf_token or None,
        device_map="auto" if use_device_map_auto else None,
    )

    # Beberapa tokenizer tidak memiliki pad_token_id. Pinjam eos_token_id agar
    # proses tokenisasi (terutama saat batching) tidak error.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Bila TIDAK pakai device_map="auto", pindahkan model ke device eksplisit.
    # (Dengan device_map="auto", model sudah diletakkan otomatis & TIDAK boleh di-.to().)
    if not use_device_map_auto:
        model = model.to(device)

    model.eval()

    trace_output("model & tokenizer siap dipakai")
    return model, tokenizer


def generate_answer(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 15,
    return_scores: bool = False,
):
    """
    Menjalankan generasi teks (greedy search) untuk SATU prompt.

    Argumen:
        prompt         : teks prompt lengkap yang akan diberikan ke model.
        tokenizer      : objek tokenizer.
        model          : objek model kausal.
        device         : device model ("cuda", "cpu", dst).
        max_new_tokens : maksimum jumlah token baru yang dihasilkan.
        return_scores  : bila True, kembalikan juga probabilitas tiap token hasil.

    Mengembalikan:
        - return_scores=False -> str (jawaban mentah dari model)
        - return_scores=True  -> tuple (jawaban: str, subtoken: list[str], skor: list[float])
    """
    # CATATAN (mode REPRODUKSI baseline): prompt MENTAH langsung di-tokenize TANPA
    # chat template — persis seperti yang dilakukan reference/ (Rafi). Chat template
    # sengaja dimatikan dulu agar hasil sebanding dengan baseline; bisa diaktifkan
    # kembali nanti untuk eksperimen lanjutan.

    # Tokenisasi prompt menjadi tensor, lalu pindahkan ke device yang sama dengan model.
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy search -> output deterministik (bisa diulang)
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Panjang token input (prompt). Token setelahnya adalah token HASIL generasi.
    input_length = inputs.input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]

    # Ubah token hasil generasi menjadi teks jawaban (tanpa skip_special_tokens,
    # mengikuti reference/ agar kolom original_answer persis sama).
    answer = tokenizer.batch_decode(generated_tokens)[0]

    if not return_scores:
        return answer

    # (Opsional) Hitung probabilitas tiap token hasil generasi.
    # Berguna untuk analisis confidence model pada tiap subtoken.
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )

    subtokens = []
    scores = []
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        subtokens.append(tokenizer.decode(tok))
        # score adalah log-probabilitas -> ubah ke probabilitas dengan exp()
        raw = score.detach().cpu().numpy() if device.startswith("cuda") else score.detach().numpy()
        scores.append(float(np.exp(raw)))

    return answer, subtokens, scores
