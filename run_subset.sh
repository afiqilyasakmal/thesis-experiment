#!/usr/bin/env bash
# =============================================================================
# run_subset.sh
# =============
# Jalankan varian BERT-similarity HANYA untuk subset model (default 3 model 7B)
# — versi ringkas dari run_experiment.sh, dipakai di Kaggle/server.
#
# Model default (semua 7B, TIDAK gated, context panjang):
#   mistralai/Mistral-7B-Instruct-v0.3
#   deepseek-ai/deepseek-llm-7b-chat
#   Qwen/Qwen2.5-7B-Instruct
#
# Pemakaian:
#   ./run_subset.sh                  # generate-bert (jika perlu) + inferensi + evaluasi
#   ./run_subset.sh generate-bert    # hanya generate 4 file contoh BERT
#   ./run_subset.sh infer            # hanya inferensi
#   ./run_subset.sh eval             # hanya evaluasi
#
# Variabel lingkungan: DEVICE, NUM_EXAMPLES, FORCE, DRY_RUN, BERT_MODEL
#
# CATATAN RESUME:
#   Inferensi menulis hasil secara inkremental ke file `.part` lalu di-rename
#   menjadi file final saat selesai. Jadi bila session mati di tengah, tinggal
#   jalankan ulang script ini — record yang sudah selesai tidak diulang.
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")"

DEVICE="${DEVICE:-cuda}"
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
BERT_MODEL="${BERT_MODEL:-indobenchmark/indobert-large-p2}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

TEST_CSV="dataset/test.csv"
TRAIN_CSV="dataset/train.csv"

SIM_FUNCTIONS=(cosine dot euclidean manhattan)

# Subset model: "repo:HuggingFace -> slug_nama_file" (dipisah ':').
MODELS=(
  "mistralai/Mistral-7B-Instruct-v0.3:mistral-7b"
  "deepseek-ai/deepseek-llm-7b-chat:deepseek-7b"
  "Qwen/Qwen2.5-7B-Instruct:qwen2.5-7b"
)

# ---------------------------------------------------------------------------
# Parsing argumen
# ---------------------------------------------------------------------------
RUN_GENERATE=0; RUN_INFER=0; RUN_EVAL=0
if [ "$#" -eq 0 ]; then
  RUN_GENERATE=1; RUN_INFER=1; RUN_EVAL=1
else
  for arg in "$@"; do
    case "$arg" in
      generate-bert|genb) RUN_GENERATE=1 ;;
      infer)              RUN_INFER=1 ;;
      eval)               RUN_EVAL=1 ;;
      all)                RUN_GENERATE=1; RUN_INFER=1; RUN_EVAL=1 ;;
      -h|--help|help)
        sed -n '2,20p' "$0"
        exit 0
        ;;
      *)
        echo "Argumen tak dikenal: $arg" >&2
        echo "Pakai: generate-bert | infer | eval | all" >&2
        exit 2
        ;;
    esac
  done
fi

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$1"; }
FAILED=()

run() {
  local desc="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then printf '  [dry-run] %s\n' "$*"; return 0; fi
  log "$desc"
  printf '  $ %s\n' "$*"
  if "$@"; then return 0; fi
  printf '\033[1;31m  !! GAGAL: %s\033[0m\n' "$desc" >&2
  FAILED+=("$desc")
  return 1
}

step() {
  local out="$1" desc="$2"; shift 2
  if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
    printf '  [skip] sudah ada: %s\n' "$out"
    return 0
  fi
  run "$desc" "$@"
}

bert_examples_file() { echo "dataset/test_with_example_bert_${1}.jsonl"; }

mkdir -p output

# ---------------------------------------------------------------------------
# Tahap 0: generate contoh few-shot BERT (4 varian similarity)
# ---------------------------------------------------------------------------
if [ "$RUN_GENERATE" = "1" ]; then
  for sim in "${SIM_FUNCTIONS[@]}"; do
    step "$(bert_examples_file "$sim")" "generate contoh few-shot BERT-${sim}" \
      python get_example/generate_examples_bert.py \
        --train_file "$TRAIN_CSV" --test_file "$TEST_CSV" \
        --num_examples "$NUM_EXAMPLES" --sim_function "$sim" \
        --bert_model "$BERT_MODEL" --output_file "$(bert_examples_file "$sim")"
  done
fi

# ---------------------------------------------------------------------------
# Tahap 1: inferensi (3 model x 4 similarity = 12 run)
# ---------------------------------------------------------------------------
if [ "$RUN_INFER" = "1" ]; then
  for sim in "${SIM_FUNCTIONS[@]}"; do
    for entry in "${MODELS[@]}"; do
      repo="${entry%%:*}"; slug="${entry##*:}"
      out="output/few_bert_${sim}_${slug}.jsonl"
      step "$out" "inferensi BERT-${sim} ${slug}" \
        python llm_inference/main.py \
          --input_file_path "$(bert_examples_file "$sim")" \
          --output_file_path "$out" \
          --prompt_type few --num_examples "$NUM_EXAMPLES" \
          --model_name "$repo" --device "$DEVICE"
    done
  done
fi

# ---------------------------------------------------------------------------
# Tahap 2: evaluasi (3 model x 4 similarity = 12 run)
# ---------------------------------------------------------------------------
if [ "$RUN_EVAL" = "1" ]; then
  for sim in "${SIM_FUNCTIONS[@]}"; do
    for entry in "${MODELS[@]}"; do
      repo="${entry%%:*}"; slug="${entry##*:}"
      in="output/few_bert_${sim}_${slug}.jsonl"
      oj="output/few_bert_${sim}_${slug}_processed.jsonl"
      oc="output/few_bert_${sim}_${slug}_report.csv"
      step "$oc" "evaluasi BERT-${sim} ${slug}" \
        python post_processing/evaluate.py \
          --input_file "$in" --output_jsonl "$oj" --output_csv "$oc" \
          --prompt_type few --model_name "$repo"
    done
  done
fi

# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------
printf '\n\033[1;32m=== SELESAI ===\033[0m\n'
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\033[1;31m  Langkah GAGAL (%d):\033[0m\n' "${#FAILED[@]}"
  for f in "${FAILED[@]}"; do printf '    - %s\n' "$f"; done
  exit 1
fi
printf '  Semua langkah berhasil.\n'
