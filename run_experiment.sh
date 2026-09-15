#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh
# ================
# Otomasi seluruh alur eksperimen klasifikasi teks 4 label (IndoNLU P2 sken 2):
#   Tahap 0  : generate contoh few-shot (generate_examples.py [random] atau
#              generate_examples_bert.py [BERT-similarity])
#   Tahap 1  : inferensi LLM (llm_inference/main.py) — 5 model x {zero, few, few-bert}
#   Tahap 2  : pembersihan + evaluasi (post_processing/evaluate.py)
#
# Pemakaian (dari folder experiment/ atau dari mana pun):
#   ./run_experiment.sh                # generate (jika perlu) + zero + few + eval
#   ./run_experiment.sh zero           # hanya zero-shot (inferensi + eval)
#   ./run_experiment.sh few            # hanya few-shot random (inferensi + eval)
#   ./run_experiment.sh bert           # hanya few-shot BERT-similarity (inferensi + eval)
#   ./run_experiment.sh infer          # hanya inferensi (zero + few random)
#   ./run_experiment.sh eval           # hanya evaluasi (zero + few random)
#   ./run_experiment.sh generate       # hanya generate contoh few-shot (random)
#   ./run_experiment.sh generate-bert  # hanya generate contoh few-shot (BERT, semua sim)
#   ./run_experiment.sh all            # random (zero + few); TANPA BERT
#   ./run_experiment.sh all bert       # random + BERT sekaligus
#
# Variabel lingkungan (opsional, bisa digabung dengan argumen di atas):
#   HF_TOKEN="hf_xxx"   token HuggingFace untuk model gated (Llama-3.1-8B)
#   DEVICE=cuda         device: cuda / cpu (default cuda)
#   NUM_EXAMPLES=5      jumlah contoh few-shot per label (default 5)
#   FORCE=1             ulang langkah walau output-nya sudah ada
#   DRY_RUN=1           hanya cetak perintah, tanpa eksekusi
#   STOP_ON_ERROR=1     berhenti total saat ada langkah gagal (default: lanjut)
# =============================================================================

set -uo pipefail

# Pastikan working directory = folder tempat script ini berada (folder experiment/),
# supaya path relatif dataset/ & output/ selalu benar dari mana pun dijalankan.
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Konfigurasi (override lewat environment variable)
# ---------------------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:-}"
DEVICE="${DEVICE:-cuda}"
NUM_EXAMPLES="${NUM_EXAMPLES:-5}"
SEED="${SEED:-42}"
BERT_MODEL="${BERT_MODEL:-indobenchmark/indobert-large-p2}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"

TEST_CSV="dataset/test.csv"
TRAIN_CSV="dataset/train.csv"
EXAMPLES_JSONL="dataset/test_with_example.jsonl"

# Fungsi similaritas untuk varian BERT (generate_examples_bert.py).
SIM_FUNCTIONS=(cosine dot euclidean manhattan)

# ---------------------------------------------------------------------------
# Pemetaan model: dua array paralel (repo HuggingFace -> slug nama file).
# Urutan sama persis dengan MODEL_LIST di config.py.
# ---------------------------------------------------------------------------
REPOS=(
  "mistralai/Mistral-7B-Instruct-v0.3"
  "meta-llama/Llama-3.1-8B-Instruct"
  "deepseek-ai/deepseek-llm-7b-chat"
  "Qwen/Qwen2.5-7B-Instruct"
  "GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"
)
SLUGS=(
  "mistral-7b"
  "llama-3.1-8b"
  "deepseek-7b"
  "qwen2.5-7b"
  "sahabatai-8b"
)

# Model yang gated (wajib --hf_token saat inferensi).
GATED_REPO="meta-llama/Llama-3.1-8B-Instruct"

usage() {
  cat <<'EOF'
Pemakaian:
  ./run_experiment.sh [tahap ...]

Tahap (boleh digabung; tanpa argumen = "all"):
  generate|gen         generate contoh few-shot RANDOM (dataset/test_with_example.jsonl)
  generate-bert|genb   generate contoh few-shot BERT (dataset/test_with_example_bert_<sim>.jsonl)
  infer                inferensi saja (zero + few random)
  zero                 zero-shot: inferensi + evaluasi
  few                  few-shot random: inferensi + evaluasi
  bert                 few-shot BERT-similarity: inferensi + evaluasi (4 sim_function)
  eval                 evaluasi saja (zero + few random)
  all                  random: generate + zero + few + eval (TANPA BERT)

Contoh gabungan:
  ./run_experiment.sh all bert     # random + BERT sekaligus

Variabel lingkungan (opsional):
  HF_TOKEN="hf_xxx"        token gated model (Llama-3.1-8B)
  DEVICE=cuda              cuda / cpu
  NUM_EXAMPLES=5           contoh per label (few-shot)
  BERT_MODEL="..."         model SentenceTransformer (default indobenchmark/indobert-large-p2)
  FORCE=1                  ulang walau output sudah ada
  DRY_RUN=1                hanya cetak perintah tanpa eksekusi
  STOP_ON_ERROR=1          berhenti saat langkah gagal
EOF
}

# ---------------------------------------------------------------------------
# Pemilihan tahap yang dijalankan
# ---------------------------------------------------------------------------
RUN_GENERATE=0; RUN_GENERATE_BERT=0
RUN_INFER_ZERO=0; RUN_INFER_FEW=0; RUN_BERT=0
RUN_EVAL_ZERO=0;  RUN_EVAL_FEW=0

if [ "$#" -eq 0 ]; then
  set -- all
fi

for arg in "$@"; do
  case "$arg" in
    all) RUN_GENERATE=1; RUN_INFER_ZERO=1; RUN_INFER_FEW=1; RUN_EVAL_ZERO=1; RUN_EVAL_FEW=1 ;;
    generate|gen)          RUN_GENERATE=1 ;;
    generate-bert|genb)    RUN_GENERATE_BERT=1 ;;
    infer)        RUN_INFER_ZERO=1; RUN_INFER_FEW=1 ;;
    zero)         RUN_INFER_ZERO=1; RUN_EVAL_ZERO=1 ;;
    few)          RUN_INFER_FEW=1;  RUN_EVAL_FEW=1 ;;
    bert)         RUN_GENERATE_BERT=1; RUN_BERT=1 ;;
    eval)         RUN_EVAL_ZERO=1;  RUN_EVAL_FEW=1 ;;
    -h|--help|help) usage; exit 0 ;;
    *) echo "Argumen tak dikenal: $arg (jalankan '$0 --help' untuk bantuan)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
FAILED=()
N_SKIP=0
INFER_TIMES=()   # akumulasi "slug|mode|start|end|durasi_detik" tiap inferensi (untuk ringkasan akhir)

log()  { printf '\n\033[1;34m=== %s ===\033[0m\n' "$1"; }
warn() { printf '\033[1;33m[perhatian]\033[0m %s\n' "$1"; }

# Format durasi (detik) menjadi "1h02m03s" / "2m05s" / "7s".
fmt_dur() {
  local s=$1 h m
  h=$((s / 3600)); m=$(((s % 3600) / 60)); s=$((s % 60))
  if [ "$h" -gt 0 ]; then printf '%dh%02dm%02ds' "$h" "$m" "$s"
  elif [ "$m" -gt 0 ]; then printf '%dm%02ds' "$m" "$s"
  else printf '%ds' "$s"; fi
}

# Jalankan satu perintah. Gagal -> dicatat, lanjut (atau berhenti bila STOP_ON_ERROR=1).
run() {
  local desc="$1"; shift
  if [ "$DRY_RUN" = "1" ]; then
    printf '  [dry-run] %s\n' "$*"
    return 0
  fi
  log "$desc"
  printf '  $ %s\n' "$*"
  if "$@"; then
    return 0
  fi
  printf '\033[1;31m  !! GAGAL: %s\033[0m\n' "$desc" >&2
  FAILED+=("$desc")
  if [ "$STOP_ON_ERROR" = "1" ]; then
    exit 1
  fi
  return 1
}

# Seperti run(), tapi lewati bila file output sudah ada (kecuali FORCE=1).
step() {
  local out="$1" desc="$2"; shift 2
  if [ -f "$out" ] && [ "$FORCE" != "1" ]; then
    printf '  [skip] sudah ada: %s\n' "$out"
    N_SKIP=$((N_SKIP + 1))
    return 0
  fi
  run "$desc" "$@"
}

# Path file contoh few-shot BERT untuk sebuah fungsi similaritas.
bert_examples_file() {
  echo "dataset/test_with_example_bert_${1}.jsonl"
}

# ---------------------------------------------------------------------------
# Ringkasan konfigurasi
# ---------------------------------------------------------------------------
printf '\033[1;32mrun_experiment.sh\033[0m — device=%s, num_examples=%s, seed=%s, bert_model=%s\n' \
  "$DEVICE" "$NUM_EXAMPLES" "$SEED" "$BERT_MODEL"
printf 'Model yang diproses:\n'
for i in "${!REPOS[@]}"; do
  printf '  - %s  (slug: %s)\n' "${REPOS[$i]}" "${SLUGS[$i]}"
done

if { [ "$RUN_INFER_ZERO" = "1" ] || [ "$RUN_INFER_FEW" = "1" ]; } && [ -z "$HF_TOKEN" ]; then
  warn "HF_TOKEN kosong. Model gated '$GATED_REPO' akan gagal unduh; set HF_TOKEN=... bila perlu."
fi

mkdir -p output

# =============================================================================
# TAHAP 0 — Generate contoh few-shot (sekali; juga dibutuhkan sebelum few-shot)
# =============================================================================
if [ "$RUN_GENERATE" = "1" ] || [ "$RUN_INFER_FEW" = "1" ]; then
  step "$EXAMPLES_JSONL" "generate contoh few-shot" \
    python get_example/generate_examples.py \
      --train_file "$TRAIN_CSV" \
      --test_file "$TEST_CSV" \
      --num_examples "$NUM_EXAMPLES" \
      --output_file "$EXAMPLES_JSONL" \
      --seed "$SEED"
fi

# Varian BERT: generate contoh few-shot berbasis similaritas semantik untuk tiap
# fungsi similaritas (cosine/dot/euclidean/manhattan).
if [ "$RUN_GENERATE_BERT" = "1" ] || [ "$RUN_BERT" = "1" ]; then
  for sim in "${SIM_FUNCTIONS[@]}"; do
    step "$(bert_examples_file "$sim")" "generate contoh few-shot BERT-${sim}" \
      python get_example/generate_examples_bert.py \
        --train_file "$TRAIN_CSV" \
        --test_file "$TEST_CSV" \
        --num_examples "$NUM_EXAMPLES" \
        --sim_function "$sim" \
        --bert_model "$BERT_MODEL" \
        --output_file "$(bert_examples_file "$sim")"
  done
fi

# =============================================================================
# TAHAP 1 — Inferensi (5 model x {zero, few})
# =============================================================================
run_inference() {
  local idx="$1" repo="$2" ptype="$3" slug="$4"
  local in out desc
  if [ "$ptype" = "zero" ]; then
    in="$TEST_CSV"
    out="output/zero_${slug}.jsonl"
  else
    in="$EXAMPLES_JSONL"
    out="output/few_${slug}.jsonl"
  fi
  desc="inferensi ${ptype}-shot (model $((idx+1))/${#REPOS[@]}): ${repo}"

  local cmd=(python llm_inference/main.py
    --input_file_path "$in"
    --output_file_path "$out"
    --prompt_type "$ptype"
    --model_name "$repo"
    --device "$DEVICE")
  if [ "$ptype" = "few" ]; then
    cmd+=(--num_examples "$NUM_EXAMPLES")
  fi
  if [ "$repo" = "$GATED_REPO" ] && [ -n "$HF_TOKEN" ]; then
    cmd+=(--hf_token "$HF_TOKEN")
  fi

  # Catat waktu mulai/selesai inferensi (diringkas jadi tabel di akhir script).
  local skipped=0 t0 t1 t0e t1e
  if [ -f "$out" ] && [ "$FORCE" != "1" ]; then skipped=1; fi
  t0=$(date '+%F %T'); t0e=$(date +%s)
  step "$out" "$desc" "${cmd[@]}"
  t1=$(date '+%F %T'); t1e=$(date +%s)
  if [ "$skipped" = "1" ]; then
    INFER_TIMES+=("$slug|$ptype|SKIP|-|-|-")
  else
    INFER_TIMES+=("$slug|$ptype|$t0|$t1|$((t1e - t0e))")
  fi
}

for i in "${!REPOS[@]}"; do
  repo="${REPOS[$i]}"; slug="${SLUGS[$i]}"
  [ "$RUN_INFER_ZERO" = "1" ] && run_inference "$i" "$repo" zero "$slug"
  [ "$RUN_INFER_FEW"  = "1" ] && run_inference "$i" "$repo" few  "$slug"
done

# =============================================================================
# TAHAP 2 — Evaluasi (5 model x {zero, few})
# =============================================================================
run_evaluate() {
  local idx="$1" repo="$2" ptype="$3" slug="$4"
  local in="output/${ptype}_${slug}.jsonl"
  local oj="output/${ptype}_${slug}_processed.jsonl"
  local oc="output/${ptype}_${slug}_report.csv"
  local desc="evaluasi ${ptype}-shot (model $((idx+1))/${#REPOS[@]}): ${repo}"

  step "$oc" "$desc" \
    python post_processing/evaluate.py \
      --input_file "$in" \
      --output_jsonl "$oj" \
      --output_csv "$oc" \
      --prompt_type "$ptype" \
      --model_name "$repo"
}

for i in "${!REPOS[@]}"; do
  repo="${REPOS[$i]}"; slug="${SLUGS[$i]}"
  [ "$RUN_EVAL_ZERO" = "1" ] && run_evaluate "$i" "$repo" zero "$slug"
  [ "$RUN_EVAL_FEW"  = "1" ] && run_evaluate "$i" "$repo" few  "$slug"
done

# =============================================================================
# TAHAP 1b & 2b — Varian BERT-similarity (few-shot): inferensi + evaluasi
# =============================================================================
run_bert_inference() {
  local idx="$1" repo="$2" sim="$3" slug="$4"
  local in out desc
  in="$(bert_examples_file "$sim")"
  out="output/few_bert_${sim}_${slug}.jsonl"
  desc="inferensi few-shot BERT-${sim} (model $((idx+1))/${#REPOS[@]}): ${repo}"

  local cmd=(python llm_inference/main.py
    --input_file_path "$in"
    --output_file_path "$out"
    --prompt_type few
    --num_examples "$NUM_EXAMPLES"
    --model_name "$repo"
    --device "$DEVICE")
  if [ "$repo" = "$GATED_REPO" ] && [ -n "$HF_TOKEN" ]; then
    cmd+=(--hf_token "$HF_TOKEN")
  fi

  step "$out" "$desc" "${cmd[@]}"
}

run_bert_evaluate() {
  local idx="$1" repo="$2" sim="$3" slug="$4"
  local in="output/few_bert_${sim}_${slug}.jsonl"
  local oj="output/few_bert_${sim}_${slug}_processed.jsonl"
  local oc="output/few_bert_${sim}_${slug}_report.csv"
  local desc="evaluasi few-shot BERT-${sim} (model $((idx+1))/${#REPOS[@]}): ${repo}"

  step "$oc" "$desc" \
    python post_processing/evaluate.py \
      --input_file "$in" \
      --output_jsonl "$oj" \
      --output_csv "$oc" \
      --prompt_type few \
      --model_name "$repo"
}

if [ "$RUN_BERT" = "1" ]; then
  for sim in "${SIM_FUNCTIONS[@]}"; do
    for i in "${!REPOS[@]}"; do
      repo="${REPOS[$i]}"; slug="${SLUGS[$i]}"
      run_bert_inference "$i" "$repo" "$sim" "$slug"
    done
  done
  for sim in "${SIM_FUNCTIONS[@]}"; do
    for i in "${!REPOS[@]}"; do
      repo="${REPOS[$i]}"; slug="${SLUGS[$i]}"
      run_bert_evaluate "$i" "$repo" "$sim" "$slug"
    done
  done
fi

# ---------------------------------------------------------------------------
# Ringkasan waktu inferensi (per model & mode) — mudah dicopy-paste
# ---------------------------------------------------------------------------
if [ "${#INFER_TIMES[@]}" -gt 0 ]; then
  printf '\n\033[1;36m=== WAKTU INFERENSI PER MODEL ===\033[0m\n'
  printf '%-16s %-6s %-20s %-20s %-10s\n' "model" "mode" "start" "end" "durasi"
  printf '%-16s %-6s %-20s %-20s %-10s\n' "-----" "----" "-----" "---" "------"
  for row in "${INFER_TIMES[@]}"; do
    IFS='|' read -r slug ptype t0 t1 dur <<< "$row"
    if [ "$t0" = "SKIP" ]; then
      printf '%-16s %-6s %s\n' "$slug" "$ptype" "SKIP (file sudah ada)"
    else
      printf '%-16s %-6s %-20s %-20s %-10s\n' \
        "$slug" "$ptype" "$t0" "$t1" "$(fmt_dur "$dur")"
    fi
  done
fi

# ---------------------------------------------------------------------------
# Ringkasan akhir
# ---------------------------------------------------------------------------
printf '\n\033[1;32m=== SELESAI ===\033[0m\n'
printf '  Langkah dilewati (output sudah ada): %d\n' "$N_SKIP"
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\033[1;31m  Langkah GAGAL (%d):\033[0m\n' "${#FAILED[@]}"
  for f in "${FAILED[@]}"; do printf '    - %s\n' "$f"; done
  exit 1
fi
printf '  Semua langkah berhasil.\n'
exit 0
