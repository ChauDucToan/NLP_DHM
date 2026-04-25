"""Tokenizer training + roundtrip test (uses a tiny corpus)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    Tokenizer,
    TokenizerConfig,
    VI2ZH_ID,
    ZH2VI_ID,
    train_sentencepiece,
)


def test_train_and_encode():
    base = [
        "Xin chào thế giới.",
        "你好，世界。",
        "Hôm nay trời đẹp.",
        "今天天气真好。",
        "Tôi yêu Việt Nam.",
        "我爱越南。",
        "Anh ấy đi học mỗi ngày.",
        "他每天去上学。",
        "Tôi muốn ăn cơm.",
        "我想吃饭。",
        "Hà Nội là thủ đô của Việt Nam.",
        "河内是越南的首都。",
        "Cô ấy đang đọc sách.",
        "她正在看书。",
    ]
    sentences = base * 200  # plenty of repetition so SP has enough to train on

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        corpus = tmp / "corpus.txt"
        corpus.write_text("\n".join(sentences), encoding="utf-8")
        train_sentencepiece(
            text_files=[corpus],
            model_prefix=tmp / "spm",
            cfg=TokenizerConfig(vocab_size=128, model_type="bpe", character_coverage=0.9995),
        )
        tok = Tokenizer(tmp / "spm.model")
        assert tok.vocab_size == 128
        assert tok.sp.PieceToId("<2vi>") == ZH2VI_ID
        assert tok.sp.PieceToId("<2zh>") == VI2ZH_ID

        # Roundtrip
        ids = tok.encode_src("你好，世界。", "zh2vi")
        assert ids[0] == ZH2VI_ID
        assert ids[-1] == EOS_ID
        out = tok.encode_tgt("Xin chào thế giới.")
        assert out[0] == BOS_ID and out[-1] == EOS_ID
        decoded = tok.decode(ids)
        assert isinstance(decoded, str)


if __name__ == "__main__":
    test_train_and_encode()
    print("Tokenizer test passed.")
