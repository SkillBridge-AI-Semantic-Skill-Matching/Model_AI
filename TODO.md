# TODO - Perbaikan skill_match & match_multi

- [ ] Update `extract_skills_matching()` di `main.py`:
  - [x] Ganti rumus skill_match agar tidak jadi 100% saat `job_skills` hanya 1 item (denom = max(job, 3))
  - [x] Tambahkan mismatch penalty berdasarkan jumlah `missing_skills`
  - [x] Cap skor maksimal (mis. 95)

- [ ] Tambahkan debug_info counts ke response `api/v1/match` dan `api/v1/match-multi`:
  - [x] job_skills_count
  - [x] cv_skills_count
  - [x] matched_skills_count
  - [x] missing_skills_count

- [ ] Perbaiki `match_multi` di `main.py`:
  - [x] Hitung match_score untuk semua job terlebih dahulu
  - [x] Simpan hasil tanpa ai_analysis
  - [x] Urutkan hasil berdasarkan `match_score` turun
  - [x] Ambil top-3 (hardcode MAX_OPENROUTER_JOBS=3)
  - [x] Generate `_generate_ai_analysis()` hanya untuk top-3
  - [x] Set ai_analysis=None untuk job di luar top-3

- [ ] Rapikan (fix) potongan kode yang tampak menempel sebelum decorator `@app.post("/api/v1/match-topk")` jika ikut terdampak edit.

- [ ] Jalankan server / uji endpoint (manual request) atau notebook yang menghasilkan tabel skor.
