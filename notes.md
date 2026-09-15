# Catatan Bug — Pemilihan Contoh BERT (fungsi similaritas euclidean/manhattan)

## Ringkasan (TL;DR)

Kode asli Alvaro (`reference/get_example/get_example_bert.py`) mengurutkan hasil
similaritas **menaik** (`[:num_of_example]`) untuk fungsi `euclidean` dan `manhattan`,
dengan asumsi `model.similarity()` mengembalikan **jarak mentah** (makin kecil =
makin mirip).

Asumsi itu **salah**. Library `sentence-transformers==3.0.0` mengembalikan **jarak
yang dinegasikan** (`euclidean_sim`/`manhattan_sim` = `-torch.cdist(...)`), sehingga
nilai **makin besar = makin mirip**. Akibatnya, pengurutan menaik justru memilih
contoh yang **paling TIDAK mirip** — kebalikan dari maksud eksperimen.

## Potensi Masalah (Bug)

Di `reference/get_example/get_example_bert.py` baris 85–91:

```python
# For cosine/dot: higher is better, for euclidean/manhattan: lower is better
if sim_function in ["cosine", "dot"]:
    top_non_porno_indices = np.argsort(non_porno_sims)[-num_of_example:][::-1]
    top_porno_indices = np.argsort(porno_sims)[-num_of_example:][::-1]
else:  # euclidean, manhattan
    top_non_porno_indices = np.argsort(non_porno_sims)[:num_of_example]
    top_porno_indices = np.argsort(porno_sims)[:num_of_example]
```

Untuk `euclidean`/`manhattan`, `np.argsort(...)` default mengurutkan **menaik**,
lalu `[:num_of_example]` mengambil nilai **terkecil**. Karena nilai yang dikembalikan
adalah `-jarak`, nilai terkecil = jarak **terbesar** = contoh **paling jauh secara
semantik**.

> Penting: *maksud* yang tertulis di skripsi ("diurutkan menaik karena jarak lebih
> kecil berarti lebih mirip") secara penalaran **benar** — tetapi hanya berlaku bila
> fungsi mengembalikan jarak mentah. Kenyataannya library mengembalikan jarak
> dinegasikan, sehingga kode melakukan kebalikan dari maksudnya.

## Bukti dari Source Code Library

Versi yang dipakai (di-pin di `reference/requirements.txt`): `sentence-transformers==3.0.0`.

1. `model.similarity()` dengan `similarity_fn_name=EUCLIDEAN`/`MANHATTAN` memetakan ke
   fungsi `*_sim`, bukan ke fungsi jarak mentah (fungsi jarak mentah bahkan tidak ada).

   `sentence_transformers/similarity_functions.py` — method `SimilarityFunction.to_similarity_fn()`:

   | similarity_fn_name | return |
   |---|---|
   | COSINE | `cos_sim` |
   | DOT_PRODUCT | `dot_score` |
   | EUCLIDEAN | `euclidean_sim` |
   | MANHATTAN | `manhattan_sim` |

2. Fungsi `*_sim` mengembalikan jarak **dinegasikan** (ada tanda minus).

   `sentence_transformers/util.py`:

   ```python
   def euclidean_sim(a, b):   return -torch.cdist(a, b, p=2.0)   # -euclidean_distance
   def manhattan_sim(a, b):   return -torch.cdist(a, b, p=1.0)   # -manhattan_distance
   ```

   Docstring-nya menegaskan: `res[i][j] = -euclidean_distance(a[i], b[j])`.

Kesimpulan: untuk **semua** empat fungsi similaritas (cosine, dot, euclidean,
manhattan), konvensi nilai yang dipakai library adalah **"makin besar = makin mirip"**.

## Contoh Numerik

Jarak mentah ke 5 kandidat (semakin kecil = semakin dekat):

```
[2, 5, 1, 8, 3]   ->  paling mirip = 1 (indeks 2)
```

Nilai yang dikembalikan `model.similarity()` (jarak dinegasikan):

```
[-2, -5, -1, -8, -3]
```

- **Kode asli (ascending)** — `np.argsort(...)[:3]`:
  urut menaik `[-8, -5, -3, -2, -1]` → ambil 3 pertama = indeks `[3, 1, 4]`
  = jarak `[8, 5, 3]` → **tiga contoh PALING JAUH**. ❌

- **Perbaikan (descending)** — `np.argsort(...)[-3:][::-1]`:
  ambil 3 terbesar = indeks `[2, 0, 4]` = jarak `[1, 2, 3]` → **tiga contoh PALING DEKAT**. ✅

## Dampak

- Hanya memengaruhi cabang `euclidean` dan `manhattan` (cabang `cosine` dan `dot`
  sudah benar).
- Pada dua cabang itu, contoh few-shot yang disuntikkan ke prompt adalah contoh yang
  **secara semantik paling tidak relevan** dengan teks uji. Hasilnya berpotensi
  lebih buruk atau tidak mencerminkan klaim metodologi ("retrieval contoh relevan"),
  dan bisa menghasilkan kesimpulan yang menyesatkan saat membandingkan fungsi
  similaritas.

## Solusi (Cara Solve)

Karena library sudah menyeragamkan konvensi "makin besar = makin mirip" untuk semua
fungsi, cukup **selalu ambil top-N nilai terbesar** (urutan menurun), tanpa percabangan
khusus untuk euclidean/manhattan.

Di `experiment/get_example/generate_examples_bert.py`:

```python
sims = similarities[label][idx]
# Ambil top-N nilai TERBESAR (paling mirip) dalam urutan menurun.
top_indices = np.argsort(sims)[-num_examples:][::-1]
```

Catatan: perbaikan ini sengaja **tidak bug-for-bug** terhadap kode Alvaro. Kalau suatu
saat ingin mereplikasi hasil Alvaro secara persis (termasuk bug-nya), tinggal kembalikan
percabangan ascending untuk euclidean/manhattan.

## Verifikasi Empiris (disarankan)

Sebelum full run, lakukan sanity-check kecil untuk memastikan arah pengurutan benar:

- Bandingkan contoh hasil `--sim_function cosine` dengan `--sim_function euclidean`
  untuk beberapa teks uji yang sama. Keduanya seharusnya menghasilkan contoh yang
  masuk akal secara semantik (topik mirip).
- Atau uji isolasi: hitung `model.similarity(emb_a, emb_b)` untuk dua teks yang
  jelas mirip vs dua teks yang jelas berbeda, lalu pastikan nilainya **lebih besar**
  untuk pasangan yang mirip pada semua fungsi similaritas.

## Referensi

- `reference/get_example/get_example_bert.py` — kode asli yang mengandung bug.
- `experiment/get_example/generate_examples_bert.py` — port yang sudah diperbaiki.
- `sentence-transformers==3.0.0`:
  - `sentence_transformers/similarity_functions.py` (mapping `to_similarity_fn`)
  - `sentence_transformers/util.py` (`euclidean_sim` / `manhattan_sim` = negated distance)
