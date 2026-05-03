# Bi-Mamba 32M — AI dịch song ngữ Trung ↔ Việt

Một mô hình dịch song ngữ **Trung → Việt** và **Việt → Trung** dùng kiến trúc
**Bi-Mamba** (selective state-space model) với khoảng **32 triệu tham số**.
Repo cung cấp **toàn bộ pipeline từ A đến Z** — tải dữ liệu, train tokenizer,
train mô hình, đánh giá SacreBLEU, dịch demo — và chạy được:

* **Trên Google Colab** end-to-end qua một notebook duy nhất:
  [`notebooks/bi_mamba_zh_vi_colab.ipynb`](notebooks/bi_mamba_zh_vi_colab.ipynb)
* **Trên máy local** (Linux + GPU CUDA hoặc CPU) bằng bộ scripts trong `scripts/`.

> Cấu trúc Bi-Mamba được chốt từ phiên bản kế hoạch ban đầu trong
> [Devin session devin-b0cfc7e955654b328d81290385130883](https://app.devin.ai/sessions/b0cfc7e955654b328d81290385130883).

---

## 1. Cấu trúc thư mục

```
bi-mamba-zh-vi/
├── README.md                     # tài liệu này
├── LICENSE                       # MIT
├── pyproject.toml                # đóng gói packages: mt_base, bi_mamba_mt, transformer_mt, hybrid_mt
├── requirements.txt              # phụ thuộc Python
├── configs/
│   ├── bi_mamba_55m.yaml             # Bi-Mamba 32.4M (model + data + train + eval)
│   ├── transformer_30m.yaml          # Transformer baseline 30.8M (cùng data + tokenizer)
│   └── hybrid_mamba_attention.yaml   # ★ Hybrid Bi-Mamba enc + Transformer dec ~32.7M
├── data/                         # nơi script ghi dữ liệu (đã .gitignore)
│   └── .gitkeep
├── notebooks/
│   ├── bi_mamba_zh_vi_colab.ipynb         # train Bi-Mamba end-to-end trên Colab
│   ├── transformer_zh_vi_colab.ipynb      # train Transformer baseline (so sánh)
│   ├── hybrid_mamba_zh_vi_colab.ipynb     # ★ train Hybrid Mamba-Attention
│   └── multi_model_zh_vi_demo.ipynb       # ★ demo linh hoạt cả 3 model trong 1 file
├── scripts/
│   ├── prepare_data.py               # tải + lọc + chia split (script-id + length filter)
│   ├── train_tokenizer.py            # train SentencePiece BPE (chia sẻ zh+vi)
│   ├── train.py                      # train Bi-Mamba
│   ├── train_transformer.py          # train Transformer baseline
│   ├── train_hybrid.py               # ★ train Hybrid Mamba-Attention
│   ├── avg_ckpts.py                  # Polyak averaging của N checkpoint cuối
│   ├── evaluate.py                   # eval Bi-Mamba (per-direction LP, length-bucket)
│   ├── evaluate_transformer.py       # eval Transformer (cùng CLI)
│   ├── evaluate_hybrid.py            # ★ eval Hybrid (strict checkpoint loading)
│   ├── sweep_decode.py               # Grid-sweep beam × LP → CSV (--model-kind {mamba,hybrid,transformer})
│   └── translate.py                  # CLI dịch (single / batch)
├── src/
│   ├── mt_base/                  # ★ Shared (tokenizer / data / trainer / eval / translate)
│   │   ├── __init__.py
│   │   ├── tokenizer.py          # SentencePiece + special tokens
│   │   ├── data.py               # Dataset, Collator, JSONL helpers, length-bucket sampler
│   │   ├── trainer.py            # generic train loop (EMA + AMP + early-stop + resume)
│   │   ├── translate.py          # generic beam-search inference
│   │   ├── evaluator.py          # SacreBLEU/chrF + length-bucket breakdown
│   │   └── utils.py              # YAML, seed, AMP dtype, device
│   ├── bi_mamba_mt/              # Bi-Mamba kiến trúc
│   │   ├── __init__.py
│   │   ├── model.py              # BiMambaTranslator (encoder + decoder)
│   │   ├── modules/              # mamba_block, bi_mamba, cross_attention, decoder_block
│   │   └── {tokenizer,data,trainer,translate,evaluator,utils}.py  # re-export shims
│   ├── transformer_mt/           # ★ Vanilla Transformer baseline
│   │   ├── __init__.py
│   │   └── model.py              # TransformerTranslator (cùng API như BiMambaTranslator)
│   └── hybrid_mt/                # ★ Hybrid Bi-Mamba enc + Transformer dec
│       ├── __init__.py
│       └── model.py              # HybridMambaAttentionTranslator (cùng API)
└── tests/
    ├── test_model.py
    ├── test_tokenizer.py
    ├── test_transformer.py
    └── test_hybrid.py
```

---

## 2. Kiến trúc

Dự án có **3 mô hình** dùng chung tokenizer + data + training loop, để ablation kiến trúc vs data:

### 2.1 Bi-Mamba (~32M tham số, v3)

| Thành phần                     | Kích thước                              |
|--------------------------------|------------------------------------------|
| Vocab (SentencePiece BPE)      | 16 000 (chia sẻ zh + vi, `character_coverage=1.0`) |
| `d_model`                      | 384                                      |
| Encoder                        | 5 × **Bi-Mamba block** + FFN(960)        |
| Decoder                        | 5 × **Mamba (causal) + Cross-attn(8 heads) + FFN(960)** |
| `d_state` / `d_conv` / expand  | 16 / 4 / 2                               |
| Tổng tham số                   | **≈ 32.4 M** (tied input + lm_head)      |

> Lịch sử kích thước: trước đây 55–63M (`d_model=512`, vocab 16–32k). v3 giảm
> xuống 32M để khớp với pool dữ liệu sạch ~135k cặp (≈ 4.2k cặp / param) —
> mô hình lớn hơn overfit lên noise và rớt BLEU.

* **Encoder Bi-Mamba**: tại mỗi block ta chạy Mamba xuôi và Mamba ngược (trên
  chuỗi đảo) rồi tổng hợp tuyến tính → ngữ cảnh hai chiều.
* **Decoder**: Mamba causal cho self-attention + multi-head cross-attention
  truy vấn ngữ cảnh nguồn → vẫn auto-regressive.
* **Bi-direction qua direction tag**: token đầu nguồn là `<2vi>` hoặc `<2zh>`
  → một mô hình duy nhất xử lý cả hai chiều dịch.

Xem `configs/bi_mamba_55m.yaml` để biết toàn bộ siêu tham số và `src/bi_mamba_mt/model.py`
cho code.

### 2.2 Vanilla Transformer baseline (~31M tham số)

| Thành phần                     | Kích thước                              |
|--------------------------------|------------------------------------------|
| Vocab (SentencePiece BPE)      | 16 000 (chia sẻ chung với Bi-Mamba)     |
| `d_model`                      | 384                                      |
| Encoder                        | 5 × MHA(8 heads) + FFN(2048), pre-norm   |
| Decoder                        | 5 × MHA + cross-attn + FFN(2048)         |
| Tổng tham số                   | **≈ 30.8 M** (tied input + lm_head)      |

Mục đích: **baseline so sánh** trên cùng data + cùng tokenizer + cùng training loop. Nếu Transformer đạt BLEU cao hơn rõ rệt → vấn đề nằm ở kiến trúc Bi-Mamba seq2seq. Nếu Transformer cũng kẹt cùng mức → vấn đề là data/tokenizer/preprocessing.

Xem `configs/transformer_30m.yaml` + `src/transformer_mt/model.py`. Train bằng `scripts/train_transformer.py`, eval bằng `scripts/evaluate_transformer.py`.

### 2.3 Hybrid Mamba-Attention (~32.7M tham số)

| Thành phần                     | Kích thước                                              |
|--------------------------------|----------------------------------------------------------|
| Vocab (SentencePiece BPE)      | 16 000 (chia sẻ chung)                                   |
| `d_model`                      | 384                                                      |
| Encoder                        | 5 × **Bi-Mamba** + FFN(960)                              |
| Decoder                        | 5 × **Transformer dec** (self + cross-attn + FFN(1536)) |
| Tổng tham số                   | **≈ 32.7 M** (tied input + lm_head)                      |

Mục đích chẩn đoán **ở đâu Bi-Mamba thua Transformer**:

* Hybrid ≫ Bi-Mamba và ≈ Transformer ⇒ decoder Mamba là bottleneck (cross-attn).
* Hybrid ≈ Bi-Mamba và ≪ Transformer ⇒ encoder Bi-Mamba yếu cho MT.
* Cả 3 ≈ nhau ⇒ bottleneck là data/tokenizer.

**Init guard:** module init không gọi `self.apply()` toàn cục — Mamba có
`dt_proj.bias`, `A_log`, `D` được init đặc biệt và sẽ bị phá nếu re-init.
`HybridMambaAttentionTranslator._init_non_mamba_weights` collect `id()` của
mọi module thuộc `MambaBlock` subtree và skip chúng. Test guard:
`tests/test_hybrid.py::test_hybrid_init_does_not_touch_mamba_internals`.

Xem `configs/hybrid_mamba_attention.yaml` + `src/hybrid_mt/model.py`. Train bằng `scripts/train_hybrid.py`, eval bằng `scripts/evaluate_hybrid.py`.

Demo nhanh cả 3 model trong 1 file: `notebooks/multi_model_zh_vi_demo.ipynb` — đổi `MODEL_KIND` ở Cell 3 (`'mamba' | 'hybrid' | 'transformer'`).

---

## 3. Dataset

Mặc định dùng [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) (config `vi-zh`, ~1M cặp).
Bạn có thể:

* Subsample để train nhanh trên Colab (`--max-train-pairs 200000`).
* Dùng dataset riêng dạng JSONL `{"zh": "…", "vi": "…"}` bằng cờ
  `--custom-jsonl path.jsonl`.
* Bổ sung VLSP / OpenSubtitles / WikiMatrix bằng cách prepend file của bạn vào
  `data/processed/train.jsonl` rồi train tokenizer lại.

---

## 4. Cài đặt local

```bash
git clone https://github.com/<user>/bi-mamba-zh-vi.git
cd bi-mamba-zh-vi
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# (Khuyên dùng nếu có GPU CUDA) — kernel selective-scan + causal-conv1d.
# `pip install` từ PyPI sẽ cố BUILD từ source và rất hay fail vì lệch ABI/CUDA;
# thay vào đó dùng wheel prebuilt trên GitHub Releases (xem ô "3b" trong
# notebook Colab). Trên local, đảm bảo `nvcc` và torch khớp rồi chạy:
pip install causal-conv1d mamba-ssm
```

> Nếu không cài được `mamba-ssm` (vd. CPU thuần, hoặc CUDA không tương thích),
> repo có **fallback PyTorch thuần** trong `mamba_block.py` — vẫn hoạt động
> đúng nhưng chậm hơn 5–10×.

Cài thêm test runner (tuỳ chọn):

```bash
pip install pytest
pytest tests/ -v
```

---

## 5. Train trên local — pipeline đầy đủ

### 5.1. Chuẩn bị dữ liệu

```bash
python scripts/prepare_data.py --config configs/bi_mamba_55m.yaml
```

Kết quả:

```
data/processed/train.jsonl
data/processed/valid.jsonl
data/processed/test.jsonl
```

Tham số hữu ích:

* `--preset {tiny|small|everyday|medium|large}` — chọn bộ corpus OPUS:
  * `tiny`     — TED2020 (~50 K, 1 MB)
  * `small`    — TED2020 + WikiMatrix + bible-uedin (~200 K) — bible chiếm ~14%, model dịch lệch giọng Kinh Thánh
  * `everyday` — (v3) TED2020 + WikiMatrix + OpenSubtitles (cap 20k) + NLLB (LASER ≥1.10, cap 20k) + bible-uedin (cap 6k) (~135 K) — **mặc định**, bible ~4%, hội thoại + đa domain, NLLB đã lọc theo điểm tin cậy
  * `medium`   — small + OpenSubtitles uncapped (~3 M) — KHÔNG khuyến nghị nếu không cap
  * `large`    — medium + NLLB (~30 M, 700 MB) — NLLB có nhiều noise pseudo-alignment
* `--sources ted2020 wikimatrix opensubtitles` — danh sách nguồn tự chọn.

**Vì sao `everyday` là mặc định (v3):** preset `small` cũ làm bible-uedin
chiếm ~14% pool sạch → model dịch "Hello, world" thành "Ngày tốt, thế
giới" (văn phong Kinh Thánh) thay vì "Xin chào, thế giới". `everyday`
giảm bible xuống ~4% qua `max_pairs_per_source.bible_uedin = 6000`, thêm
OpenSubtitles (cap 20k) cho hội thoại đời thường, và NLLB (cap 20k với
LASER ≥ 1.10) cho đa domain — chỉ giữ phần đầu high-confidence của file
score sidecar.

> **Lịch sử của `everyday` preset (3 lần lặp):**
>
> * **v1** (NLLB cap 50k random + OpenSubtitles cap 80k → ~225K cặp) → BLEU
>   zh→vi 5.96 vs baseline 47.89 (`small`). NLLB pseudo-alignment + OS fragments
>   nhiễu tràn ngập signal.
> * **v2** (bỏ NLLB hoàn toàn, OS cap 20k → ~105K) → BLEU vẫn ~6–9, do data
>   thu hẹp quá mức và tokenizer vẫn drop ký tự CJK hiếm (UNK `⁇` xuất hiện ở
>   output vi→zh).
> * **v3** (hiện tại): NLLB **filter theo LASER score ≥ 1.10** + cap 20k →
>   ~135K cặp; tokenizer `character_coverage=1.0`; mô hình giảm xuống 32M
>   params để tương xứng với cỡ dữ liệu.

**Chuẩn hoá phương ngữ tiếng Trung (mới):** các nguồn OPUS zh-vi không cùng
một thứ tiếng — TED2020.vi-zh.zh thực ra là **tiếng Quảng Đông viết phồn
thể** (~70% câu chứa hạt từ Cantonese 嘅/哋/啲/咁/喺/嗰/咗/佢...),
bible-uedin là tiếng Trung phồn thể, WikiMatrix lẫn lộn Giản/Phồn/Cổ văn,
chỉ OpenSubtitles vi-zh_cn là Mandarin Giản thể đúng nghĩa. Pipeline mới
áp dụng:

* `data.zh_filter_cantonese: true` — bỏ các cặp có hạt từ Cantonese.
* `data.zh_normalize_simplified: true` — dùng OpenCC chuyển Trad→Simp ở
  phía zh, để toàn bộ pool nhất quán Mandarin Giản thể.

Sau khi áp filter + normalize + score-filter, TED2020 còn ~3k cặp (sạch
Mandarin), WikiMatrix giữ ~85k, OpenSubtitles 20k, NLLB 20k (chỉ phần
LASER ≥ 1.10), bible 6k → tổng ~135k pairs sạch + nhất quán phương ngữ.

> NLLB ships kèm file `NLLB.vi-zh.scores` (LASER cosine similarity, sort
> giảm dần). Cấu hình `data.min_score_per_source.nllb = 1.10` chỉ giữ
> phần đầu high-confidence (~10% pool). Nâng lên 1.20 cho cleaner pool ở
> mức ~1–2%; hạ xuống 1.05 cho thêm data nhưng nhiễu hơn. Để dùng NLLB
> không filter (kiểu cũ), bỏ entry `nllb` ra khỏi `min_score_per_source`.
> (Lưu ý: `min_score_per_source` áp dụng cho mọi preset có chứa NLLB,
> kể cả `large` — chỉ thay preset không tắt được score filter.)

**Lưu ý quan trọng về `medium` / `large` không cap**: OpenSubtitles vi-zh_cn
(chiếm ~85% medium) và NLLB là parallel pseudo-aligned, có rất nhiều noise
và phong cách hội thoại ngắn. Khi dùng nguyên cục, nó nuốt gọn dữ liệu
sạch → test set cũng bị nhiễm noise → BLEU collapse từ ~16 xuống ~2. Nên
**cap chúng** qua `data.max_pairs_per_source.opensubtitles: 80000`
(mặc định) để có data đa dạng mà vẫn giữ được signal sạch.
* `--max-train-pairs 200000` — subsample tập train.
* `--custom-jsonl my.jsonl` — dùng dataset riêng (mỗi dòng `{"zh": "...", "vi": "..."}`).

### 5.2. Train tokenizer (SentencePiece BPE chung)

```bash
python scripts/train_tokenizer.py --config configs/bi_mamba_55m.yaml
```

Output: `data/tokenizer/spm.model`, `spm.vocab`.

### 5.3. Train mô hình

```bash
python scripts/train.py --config configs/bi_mamba_55m.yaml
```

Trong quá trình train sẽ in `loss`, `lr`, `tok/s` mỗi `log_every` step. Mỗi
`save_every` step sẽ lưu `runs/bi_mamba_55m/checkpoint_step{N}.pt` và
`runs/bi_mamba_55m/latest.pt`.

Resume:

```bash
python scripts/train.py --config configs/bi_mamba_55m.yaml \
    --resume runs/bi_mamba_55m/latest.pt
```

`latest.pt` / `checkpoint_step{N}.pt` lưu cả model, optimizer, GradScaler,
EMA, global step và early-stopping state, nên resume sẽ tiếp tục đúng lịch LR
thay vì khởi động lại optimizer/scheduler từ đầu. Dùng `latest_ema.pt` cho
evaluate/inference, không dùng làm checkpoint resume chính.

**Early stopping** (mặc định bật): nếu `ema_val_loss` (hoặc `val_loss` khi
`ema=False`) không cải thiện qua `early_stopping_patience` lần eval liên
tiếp (mặc định 6K steps với `eval_every=1000`), training tự dừng và lưu checkpoint
cuối. Set `early_stopping_patience: 0` trong config để tắt.

### 5.4. Best / Average / EMA checkpoint (boost BLEU "miễn phí")

Trainer ghi nhiều checkpoint khác nhau:

* `latest.pt` / `latest_ema.pt` — weights ở step cuối cùng.
* `best.pt` — weights ở step có **`val_loss` thấp nhất**.
* `best_ema.pt` — EMA weights ở step có **`ema_val_loss` thấp nhất**.

Để ép thêm BLEU mà không cần train tiếp, average **5 checkpoint cuối**
(Polyak averaging) — cả raw và EMA:

```bash
python scripts/avg_ckpts.py --ckpts-dir runs/bi_mamba_55m --n 5
python scripts/avg_ckpts.py --ckpts-dir runs/bi_mamba_55m --n 5 --ema
```

Sáu checkpoint thường được so sánh: `latest.pt`, `latest_ema.pt`, `best.pt`,
`best_ema.pt`, `avg_last5.pt`, `avg_last5_ema.pt`. Với run đủ dài thì
`avg_last5_ema.pt` thường thắng; với run ngắn / overfit thì `best_ema.pt`
thắng. EMA + averaging cho +0.5–2 BLEU gần như miễn phí.

### 5.5. Đánh giá SacreBLEU + chrF

```bash
python scripts/evaluate.py --config configs/bi_mamba_55m.yaml \
    --checkpoint runs/bi_mamba_55m/best_ema.pt --num-samples 5000 --beam-size 4
```

`length_penalty` được đọc từ config theo từng chiều
(`zh2vi: 1.00`, `vi2zh: 0.90` mặc định). Override bằng
`--length-penalty 1.0` (áp cả hai chiều) hoặc
`--length-penalty-zh2vi` / `--length-penalty-vi2zh` (từng chiều) nếu muốn.

Thêm `--length-buckets` để in BLEU/chrF theo bucket độ dài nguồn
(`short: <20`, `medium: 20–50`, `long: ≥50` ký tự):

```bash
python scripts/evaluate.py --config configs/bi_mamba_55m.yaml \
    --checkpoint runs/bi_mamba_55m/best_ema.pt \
    --num-samples 5000 --beam-size 4 --length-buckets
```

### 5.5b. Grid-sweep decoding → CSV

Để tìm `(beam, length_penalty)` tối ưu mà không phải chạy
`evaluate.py` thủ công nhiều lần:

```bash
python scripts/sweep_decode.py \
    --config configs/bi_mamba_55m.yaml \
    --checkpoint runs/bi_mamba_55m/best_ema.pt \
    --num-samples 2000 \
    --beams 1 2 4 6 \
    --lp-zh2vi 0.8 0.9 1.0 1.1 1.2 \
    --lp-vi2zh 0.6 0.8 0.9 1.0 \
    --out runs/bi_mamba_55m/sweep.csv
```

Grid chạy độc lập cho mỗi chiều (`beam × lp_zh2vi` cho zh→vi,
`beam × lp_vi2zh` cho vi→zh) nên không bị blowup cartesian.
CSV có cột `direction,beam,length_penalty,bucket,n,bleu,chrf`;
thêm `--length-buckets` để đính kèm cả dòng per-bucket.
Checkpoint chỉ load **một lần**.

### 5.6. Dịch

```bash
# Một câu
python scripts/translate.py --config configs/bi_mamba_55m.yaml --direction zh2vi --text "你好，世界！"

# File (mỗi dòng một câu)
python scripts/translate.py --config configs/bi_mamba_55m.yaml --direction vi2zh \
    --input my_vi.txt --output my_zh.txt --beam-size 4
```

---

## 6. Train end-to-end trên Google Colab

Mở [`notebooks/bi_mamba_zh_vi_colab.ipynb`](notebooks/bi_mamba_zh_vi_colab.ipynb) — chỉ cần chạy
tuần tự các ô. Notebook sẽ:

1. Mount Drive (tuỳ chọn).
2. Clone repo.
3. Cài deps cơ bản, rồi (3b) cài CUDA fast-path từ GitHub Releases — auto-detect torch/CUDA/Python để tải đúng wheel `mamba-ssm` + `causal-conv1d`. Fail thì silent fallback sang pure-PyTorch.
4. Tải corpus zh-vi từ OPUS (TED2020 + WikiMatrix + bible-uedin theo mặc định).
5. Train tokenizer.
6. Train mô hình (mặc định 200K cặp × 30K steps trên T4 ≈ 1.5–2 giờ).
7. Đánh giá BLEU/chrF cả hai chiều.
8. Demo dịch một số câu mẫu.
9. Lưu checkpoint sang Drive.

---

## 7. Tham số ước tính

* **Thời gian train (T4, AMP bf16, batch 32, 30K steps, 200K cặp):** ~ 1.5–2 giờ.
* **Thời gian train (A100, batch 128, 60K steps, 1M cặp):** ~ 3–5 giờ.
* **VRAM:** ~ 6–8 GB ở batch 32 / max_len 256.
* **BLEU dự kiến** (200K cặp, 30K steps): zh → vi 18–24, vi → zh 16–22.
  Với toàn corpus + 60K steps có thể đạt 26–32 / 24–30.

---

## 8. Tuỳ biến

* **Đổi kích thước mô hình:** chỉnh `d_model`, `n_encoder_layers`, `n_decoder_layers`,
  `d_ff`, `expand` trong `configs/bi_mamba_55m.yaml`. Chạy lại training là đủ.
* **Thêm dữ liệu của bạn:** đặt JSONL `{zh, vi}` của bạn rồi
  `prepare_data.py --custom-jsonl ...`.
* **Tách monolingual data để pre-train decoder:** mở rộng `data.py` để hỗ trợ
  monolingual nếu cần.
* **Dùng kernel CUDA chính chủ:** cài `mamba-ssm` + `causal-conv1d`. Repo tự
  detect và switch sang fast-path.

---

## 9. Tests

```bash
pytest tests/ -v
```

* `tests/test_model.py` — kiểm tra forward, backward, sự nhất quán giữa
  full-sequence scan và step-by-step decode, và greedy generate.
* `tests/test_tokenizer.py` — train SentencePiece trên một mini corpus và
  kiểm tra special tokens + roundtrip.

---

## 10. Trích dẫn / Tham khảo

* Gu & Dao, **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**, 2023.
  <https://arxiv.org/abs/2312.00752>
* OPUS: <https://opus.nlpl.eu/> — TED2020, WikiMatrix, bible-uedin, OpenSubtitles, NLLB
* `mamba-ssm`: <https://github.com/state-spaces/mamba>

---

## License

MIT — xem [LICENSE](LICENSE).
