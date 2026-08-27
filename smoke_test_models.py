"""
smoke_test_models.py
====================
Uji cepat: memastikan SEMUA model di config.MODEL_LIST bisa diunduh, dimuat ke
GPU, dan menghasilkan output singkat. Script ini memakai jalur yang sama persis
dengan inferensi asli (load_model_tokenizer + generate_answer), jadi kalau ada
model yang bermasalah (gated/403, OOM, dll.) akan ketahuan di sini SEBELUM
full run.

Cara pakai (dari root proyek = folder experiment/):
    python smoke_test_models.py
    python smoke_test_models.py --device cuda --dtype bfloat16
    HF_TOKEN="hf_xxx" python smoke_test_models.py
    python smoke_test_models.py --hf_token hf_xxx
"""

import argparse
import gc
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import DEFAULT_DEVICE, DEFAULT_DTYPE, MODEL_LIST
from llm_inference.model_utils import generate_answer, load_model_tokenizer


def _vram_gb() -> float:
    """Total memori GPU teralokasi (semua device), dalam GB."""
    if not torch.cuda.is_available():
        return 0.0
    return sum(torch.cuda.memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test semua model (load + generate).")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""),
                        help="Token HF (opsional; bisa juga lewat env var HF_TOKEN).")
    parser.add_argument("--prompt", type=str, default="Sebutkan satu kata dalam bahasa Indonesia.")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    args = parser.parse_args()

    print(f"device={args.device} | dtype={args.dtype} | max_new_tokens={args.max_new_tokens}")
    if torch.cuda.is_available():
        print(f"CUDA tersedia: {torch.cuda.device_count()} GPU")
    else:
        print("CUDA TIDAK tersedia (model akan dimuat ke CPU)")

    hasil = []
    for i, model_name in enumerate(MODEL_LIST, 1):
        print(f"\n=== [{i}/{len(MODEL_LIST)}] {model_name} ===")
        model = tokenizer = None
        try:
            model, tokenizer = load_model_tokenizer(
                model_name, args.hf_token, args.device, args.dtype
            )
            answer = generate_answer(
                args.prompt, tokenizer, model, args.device,
                max_new_tokens=args.max_new_tokens, return_scores=False,
            )
            print(f"  OK     -> jawaban: {answer.strip()[:80]!r}")
            print(f"  VRAM terpakai: {_vram_gb():.2f} GB")
            hasil.append((model_name, True, ""))
        except Exception as e:
            print(f"  GAGAL  -> {type(e).__name__}: {e}")
            hasil.append((model_name, False, str(e)))
        finally:
            # Bebaskan memori sebelum pindah ke model berikutnya.
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("RINGKASAN")
    print("=" * 60)
    for model_name, ok, err in hasil:
        status = "OK" if ok else "GAGAL"
        line = f"  [{status}] {model_name}"
        if not ok:
            line += f"  -> {err}"
        print(line)
    n_ok = sum(1 for _, ok, _ in hasil if ok)
    print(f"\n{n_ok}/{len(hasil)} model berjalan.")


if __name__ == "__main__":
    main()
