# Diagram Alir Penelitian

Diagram alir (Mermaid) untuk dua eksperimen klasifikasi konten pornografi teks
bahasa Indonesia (IndoNLU P2 — **Skenario 2, 4 label**):

- **Eksperimen 1** — mengikuti metode **Rafi Madani** (pemilihan contoh *random*).
- **Eksperimen 2** — mengikuti metode **Alvaro Luqman** (pemilihan contoh berbasis
  *similaritas semantik* Sentence-BERT).

Keduanya memakai alur 3 tahap yang sama dengan `reference/`:

> **generate contoh → inferensi LLM → evaluasi**

Satu-satunya perbedaan ada di **Tahap 0 (cara memilih contoh few-shot)**.

Diagram ini dirender oleh GitHub, VSCode (dengan ekstensi Mermaid), dan Kaggle.

---

## Eksperimen 1 — Metode Rafi (random) → Skenario 2 (4 label)

```mermaid
flowchart LR
    subgraph DATA["Data — Skenario 2 (4 label)"]
        TRAIN[("dataset/train.csv<br/>6.130 data")]
        TEST[("dataset/test.csv<br/>2.044 data")]
    end

    subgraph S0["Tahap 0 — Generate contoh (Rafi: random)"]
        GEN["get_example/generate_examples.py<br/>random.sample, K=5 per kelas"]
        EX[("dataset/test_with_example.jsonl<br/>contoh acak per baris")]
    end

    TRAIN --> GEN
    TEST --> GEN
    GEN --> EX

    subgraph S1["Tahap 1 — Inferensi LLM (5 model)"]
        MN["llm_inference/main.py<br/>greedy · max_new_tokens=15 · bfloat16"]
        Z[("output/zero_*.jsonl<br/>hasil zero-shot")]
        F[("output/few_*.jsonl<br/>hasil few-shot")]
    end

    TEST -->|"zero-shot"| MN
    EX -->|"few-shot"| MN
    MN --> Z
    MN --> F

    subgraph S2["Tahap 2 — Evaluasi"]
        EV["post_processing/evaluate.py<br/>normalize_answer + metrik"]
        RP[("report.csv<br/>+ confusion matrix")]
    end

    Z --> EV
    F --> EV
    EV --> RP
```

### Keterangan Eksperimen 1

| Bagian | Detail |
|---|---|
| Data | 4 label: `non_porno`, `percakapan_porno`, `penyebar_konten_porno`, `penjaja_seks` |
| Seleksi contoh | `random.sample`, K=5 per kelas (mengikuti `reference/get_example/get_example_random.py`) |
| Model | 5 LLM: Mistral-7B, Llama-3.1-8B, DeepSeek-7B, Qwen2.5-7B, SahabatAI-8B |
| Mode prompt | zero-shot (input `test.csv`) + few-shot (input `test_with_example.jsonl`) |
| Decoding | greedy, `max_new_tokens=15`, `bfloat16` |
| Evaluasi | Macro-F1 (+ precision/recall/F1 per kelas, confusion matrix) |

---

## Eksperimen 2 — Metode Alvaro (SBERT) → Skenario 2 (4 label)

```mermaid
flowchart LR
    subgraph DATA["Data — Skenario 2 (4 label)"]
        TRAIN[("dataset/train.csv<br/>6.130 data")]
        TEST[("dataset/test.csv<br/>2.044 data")]
    end

    subgraph S0["Tahap 0 — Generate contoh (Alvaro: SBERT)"]
        SBERT["Sentence-BERT<br/>indobenchmark/indobert-large-p2"]
        SIM["4 fungsi similaritas<br/>cosine · dot · euclidean · manhattan"]
        GEN["get_example/generate_examples_bert.py<br/>top-K=5 per kelas paling mirip"]
        EX[("dataset/test_with_example_bert_*.jsonl<br/>4 file, satu per fungsi")]
    end

    TRAIN --> SBERT
    TEST --> SBERT
    SBERT --> SIM
    SIM --> GEN
    GEN --> EX

    subgraph S1["Tahap 1 — Inferensi LLM (4 model)"]
        MN["llm_inference/main.py<br/>greedy · max_new_tokens=15 · bfloat16"]
        F[("output/few_bert_*.jsonl<br/>hasil few-shot per fungsi & model")]
    end

    EX -->|"few-shot"| MN
    MN --> F

    subgraph S2["Tahap 2 — Evaluasi"]
        EV["post_processing/evaluate.py<br/>normalize_answer + metrik"]
        RP[("report.csv<br/>+ confusion matrix")]
    end

    F --> EV
    EV --> RP
```

### Keterangan Eksperimen 2

| Bagian | Detail |
|---|---|
| Data | Sama dengan Eksperimen 1 (4 label) |
| Seleksi contoh | Sentence-BERT `indobert-large-p2`, 4 fungsi similaritas (cosine, dot, euclidean, manhattan), K=5 per kelas paling mirip |
| Model | 4 LLM (Alvaro: tanpa DeepSeek): Mistral-7B, Llama-3.1-8B, Qwen2.5-7B, SahabatAI-8B |
| Mode prompt | few-shot saja (4 fungsi similaritas × model) |
| Decoding | greedy, `max_new_tokens=15`, `bfloat16` |
| Evaluasi | Macro-F1 (+ perbandingan antar fungsi similaritas) |

---

## Perbedaan Kunci Eksperimen 1 vs 2

| | Eksperimen 1 (Rafi) | Eksperimen 2 (Alvaro) |
|---|---|---|
| Seleksi contoh | Random (statis, acak) | SBERT (dinamis, paling mirip semantik) |
| Varian | zero + few | few saja, 4 fungsi similaritas |
| Model | 5 (termasuk DeepSeek) | 4 (tanpa DeepSeek) |
| Output Tahap 0 | 1 file JSONL | 4 file JSONL (satu per fungsi) |
