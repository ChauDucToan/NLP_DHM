# Bi-Mamba ~110M — AI dịch song ngữ Trung ↔ Việt

Một mô hình dịch song ngữ **Trung → Việt** và **Việt → Trung** dùng kiến trúc
**Bi-Mamba** (selective state-space model) với khoảng **110 triệu tham số**
(d_model=640, layers 6/6, vocab=32k). Phiên bản cũ 55M (d_model=512, layers
5/5) vẫn chạy được bằng cách override `d_model` v.v. trong config.
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
├── pyproject.toml                # đóng gói package bi_mamba_mt
├── requirements.txt              # phụ thuộc Python
├── configs/
│   └── bi_mamba_55m.yaml         # config 55M (model + tokenizer + data + train + eval)
├── data/                         # nơi script ghi dữ liệu (đã .gitignore)
│   └── .gitkeep
├── notebooks/
│   └── bi_mamba_zh_vi_colab.ipynb  # train end-to-end trên Colab
├── scripts/
│   ├── prepare_data.py           # tải + lọc + chia split (script-id + length filter)
│   ├── train_tokenizer.py        # train SentencePiece BPE (chia sẻ zh+vi)
│   ├── train.py                  # vòng lặp train chính (EMA + length-bucket sampler)
│   ├── avg_ckpts.py              # Polyak averaging của N checkpoint cuối
│   ├── evaluate.py               # SacreBLEU + chrF cả hai chiều, per-direction LP
│   └── translate.py              # CLI dịch (single / batch)
├── src/
│   └── bi_mamba_mt/              # package Python
│       ├── __init__.py
│       ├── model.py              # BiMambaTranslator (encoder + decoder)
│       ├── tokenizer.py          # SentencePiece + special tokens
│       ├── data.py               # Dataset, Collator, JSONL helpers
│       ├── trainer.py            # train loop, optimizer, AMP, lr schedule
│       ├── translate.py          # encode → generate → decode
│       ├── evaluator.py          # SacreBLEU/chrF
│       ├── utils.py              # YAML, seed, AMP dtype, device
│       └── modules/
│           ├── mamba_block.py    # Mamba selective SSM (CUDA fast path + PyTorch fallback)
│           ├── bi_mamba.py       # Bi-Mamba (forward + reversed)
│           ├── cross_attention.py
│           └── decoder_block.py
└── tests/
    ├── test_model.py
    └── test_tokenizer.py
```

---

## 2. Kiến trúc Bi-Mamba (~110M tham số, default)

| Thành phần                     | Kích thước                              |
|--------------------------------|------------------------------------------|
| Vocab (SentencePiece BPE)      | 32 000 (chia sẻ zh + vi)                 |
| `d_model`                      | 640                                      |
| Encoder                        | 6 × **Bi-Mamba block** + FFN(1792)       |
| Decoder                        | 6 × **Mamba (causal) + Cross-attn(10 heads, head_dim 64) + FFN(1792)** |
| `d_state` / `d_conv` / expand  | 16 / 4 / 2                               |
| Tổng tham số                   | **≈ 110 M** (tied input + lm_head)       |
| **Regularization**             | EMA(0.999) + R-Drop(α=1) + BPE-dropout(α=0.1) |
| **Decoding**                   | beam=6, per-direction LP (zh→vi: 1.20, vi→zh: 0.80), multi-checkpoint ensemble |

* **Encoder Bi-Mamba**: tại mỗi block ta chạy Mamba xuôi và Mamba ngược (trên
  chuỗi đảo) rồi tổng hợp tuyến tính → ngữ cảnh hai chiều.
* **Decoder**: Mamba causal cho self-attention + multi-head cross-attention
  truy vấn ngữ cảnh nguồn → vẫn auto-regressive.
* **Bi-direction qua direction tag**: token đầu nguồn là `<2vi>` hoặc `<2zh>`
  → một mô hình duy nhất xử lý cả hai chiều dịch.

Xem `configs/bi_mamba_55m.yaml` để biết toàn bộ siêu tham số và `src/bi_mamba_mt/model.py`
cho code.

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

* `--preset {tiny|small|medium|large}` — chọn bộ corpus OPUS:
  * `tiny`   — TED2020 (~50 K, 1 MB)
  * `small`  — TED2020 + WikiMatrix + bible-uedin (~200 K, 25 MB) — **mặc định, BLEU cao nhất**
  * `medium` — small + OpenSubtitles vi-zh_cn (~3 M, 65 MB) — chỉ dùng nếu cap OpenSubtitles
  * `large`  — medium + NLLB (~30 M, 700 MB)
* `--sources ted2020 wikimatrix opensubtitles` — danh sách nguồn tự chọn.

**Lưu ý quan trọng về `medium` / `large`**: OpenSubtitles vi-zh_cn (chiếm
~85% medium) và NLLB là parallel pseudo-aligned, có rất nhiều noise và
phong cách hội thoại ngắn. Khi dùng nguyên cục, nó nuốt gọn dữ liệu sạch
→ test set cũng bị nhiễm noise → BLEU collapse từ ~16 xuống ~2. Nên
**cap chúng** qua `data.max_pairs_per_source.opensubtitles: 100000` (mặc
định trong config) để có data đa dạng mà vẫn giữ được signal sạch.
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

**Early stopping** (mặc định bật): nếu `ema_val_loss` (hoặc `val_loss` khi
`ema=False`) không cải thiện qua `early_stopping_patience=10` lần eval liên
tiếp (≈ 20K steps với `eval_every=2000`), training tự dừng và lưu checkpoint
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
    --checkpoint runs/bi_mamba_55m/best_ema.pt --num-samples 1000
```

`length_penalty` được đọc từ config theo từng chiều
(`zh2vi: 1.20`, `vi2zh: 0.80` mặc định). Override bằng
`--length-penalty 1.0` nếu muốn.

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
