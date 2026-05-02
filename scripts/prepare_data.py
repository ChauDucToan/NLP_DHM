"""Download + clean + split the parallel zh-vi corpus.

By default this downloads parallel data **directly from OPUS** (the
``object.pouta.csc.fi`` mirror) — no Hugging Face dataset config needed,
since OPUS-100 does not actually contain a zh-vi pair.

Available presets (chosen via ``--preset``):

* ``tiny``      — TED2020 only (~ 50 k pairs, ~ 1 MB).         Good for smoke tests.
* ``small``     — TED2020 + WikiMatrix + bible-uedin
                  (~ 200 k pairs, ~ 25 MB).                    Bible-heavy (~14%).
* ``everyday``  — TED2020 + WikiMatrix + OpenSubtitles + bible-uedin (capped)
                  (~ 200 k pairs, ~ 65 MB).                    **Default.**
                  Bible only ~3% of mix, OpenSubtitles adds conversational
                  vocabulary (everyday phrases, greetings) so the model can
                  translate phrases like "hello world" naturally.
* ``medium``    — small + OpenSubtitles vi-zh_cn (~ 3 M pairs, ~ 65 MB zip).
* ``large``     — medium + NLLB / CCMatrix (~ 30 M pairs).     For full-corpus runs.

Note: other zh-vi corpora on OPUS (Tatoeba, QED, ALT, NeuLab-TedTalks,
wikimedia) are either unavailable or have a corrupted Chinese side
(many characters dropped during alignment) and are intentionally excluded.

You can also use ``--custom-jsonl path/to/your.jsonl`` to skip downloads
and use your own corpus, where each line is ``{"zh": "...", "vi": "..."}``.

Run::

    python scripts/prepare_data.py --config configs/bi_mamba_55m.yaml --preset small
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from urllib.request import Request, urlopen

# Make ``src/`` importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bi_mamba_mt.data import Pair, basic_clean, pair_ok, write_jsonl
from bi_mamba_mt.utils import load_yaml

# Optional Traditional→Simplified normalization (recommended for zh-vi corpora
# because OPUS zh-vi sources are heterogeneous: TED2020 is Cantonese in
# Traditional script, bible-uedin is Traditional, WikiMatrix is mixed,
# OpenSubtitles vi-zh_cn is Simplified). Without normalization the model is
# trained on 4 different language varieties on the zh side which produces
# stylistically inconsistent translations.
try:
    import opencc  # type: ignore
    _OPENCC_T2S = opencc.OpenCC("t2s")
except ImportError:  # pragma: no cover
    _OPENCC_T2S = None


# Cantonese-only characters (effectively absent from Mandarin written text).
# Pairs whose zh side contains any of these are dropped — they are written
# in Cantonese (粵語), not Mandarin, and would teach the model the wrong
# language. Curated list:
#   嘅 (possessive de), 哋 (plural marker), 啲 (some / a bit), 咁 (so / such),
#   喺 (at / in), 嗰 (that), 咗 (perfective le), 佢 (3rd-person pronoun),
#   嚟 (come / lai), 㗎 (emphasis particle), 噉 (so / like that), 嘞 (modal).
_CANTONESE_PARTICLES = frozenset("嘅哋啲咁喺嗰咗佢嚟㗎噉嘞")


def _looks_cantonese(zh: str) -> bool:
    return any(c in _CANTONESE_PARTICLES for c in zh)


def _normalize_zh(zh: str) -> str:
    """Convert Traditional Chinese to Simplified, leaving Simplified untouched."""
    if _OPENCC_T2S is None:
        return zh
    return _OPENCC_T2S.convert(zh)


# ---------------------------------------------------------------------
# OPUS sources
# ---------------------------------------------------------------------
@dataclass
class OpusSource:
    name: str           # human label
    url: str            # zip URL
    vi_member: str      # filename of Vietnamese side inside zip
    zh_member: str      # filename of Chinese side inside zip


# Only sources whose URLs return HTTP 200 on the Pouta mirror.
SOURCES: dict[str, OpusSource] = {
    "ted2020": OpusSource(
        name="TED2020 vi-zh",
        url="https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/vi-zh.txt.zip",
        vi_member="TED2020.vi-zh.vi",
        zh_member="TED2020.vi-zh.zh",
    ),
    "wikimatrix": OpusSource(
        name="WikiMatrix vi-zh",
        url="https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/vi-zh.txt.zip",
        vi_member="WikiMatrix.vi-zh.vi",
        zh_member="WikiMatrix.vi-zh.zh",
    ),
    "bible_uedin": OpusSource(
        name="bible-uedin vi-zh",
        url="https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/vi-zh.txt.zip",
        vi_member="bible-uedin.vi-zh.vi",
        zh_member="bible-uedin.vi-zh.zh",
    ),
    "opensubtitles": OpusSource(
        name="OpenSubtitles vi-zh_cn",
        url="https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/vi-zh_cn.txt.zip",
        vi_member="OpenSubtitles.vi-zh_cn.vi",
        zh_member="OpenSubtitles.vi-zh_cn.zh_cn",
    ),
    "nllb": OpusSource(
        name="NLLB vi-zh",
        url="https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/vi-zh.txt.zip",
        vi_member="NLLB.vi-zh.vi",
        zh_member="NLLB.vi-zh.zh",
    ),
    "ccmatrix": OpusSource(
        name="CCMatrix vi-zh",
        url="https://object.pouta.csc.fi/OPUS-CCMatrix/v1/moses/vi-zh.txt.zip",
        vi_member="CCMatrix.vi-zh.vi",
        zh_member="CCMatrix.vi-zh.zh",
    ),
}

PRESETS: dict[str, list[str]] = {
    "tiny":     ["ted2020"],
    "small":    ["ted2020", "wikimatrix", "bible_uedin"],
    # `everyday` is the recommended default: bible-uedin gets a tight cap (~3%
    # of the mix) so the model is not biased toward biblical register, and
    # OpenSubtitles is included (capped) for conversational vocabulary.
    "everyday": ["ted2020", "wikimatrix", "opensubtitles", "bible_uedin"],
    "medium":   ["ted2020", "wikimatrix", "bible_uedin", "opensubtitles"],
    "large":    ["ted2020", "wikimatrix", "bible_uedin", "opensubtitles", "nllb"],
}


# ---------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------
def download(url: str, dest: Path, timeout: int = 300) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cache hit:   {dest}")
        return
    print(f"  downloading: {url}")
    req = Request(url, headers={"User-Agent": "bi-mamba-mt/0.1"})
    with urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        total = int(r.headers.get("content-length", 0))
        read = 0
        last_pct = -10
        chunk = 1024 * 1024
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = int(100 * read / total)
                if pct >= last_pct + 5:
                    print(f"    {pct:3d}%  ({read/1e6:.1f} / {total/1e6:.1f} MB)")
                    last_pct = pct
    print(f"  saved:       {dest}  ({dest.stat().st_size/1e6:.1f} MB)")


def iter_pairs_from_opus_zip(
    zip_path: Path,
    src: OpusSource,
    *,
    normalize_zh: bool = True,
    filter_cantonese: bool = True,
) -> Iterable[Pair]:
    """Yield ``Pair`` rows from an OPUS Moses-format zip.

    ``normalize_zh`` runs Traditional→Simplified normalisation on the zh side
    using OpenCC. ``filter_cantonese`` drops pairs whose zh side contains
    Cantonese-only particles (嘅/哋/啲/咁/喺/嗰/咗) — these are typically
    Cantonese sentences mis-labelled as zh on OPUS (notably TED2020.vi-zh).
    """
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        if src.vi_member not in names or src.zh_member not in names:
            print(
                f"  WARNING: expected members not found in {zip_path.name}; got {sorted(names)}"
            )
            return
        with z.open(src.vi_member) as fv, z.open(src.zh_member) as fz:
            for vi_line, zh_line in zip(
                io.TextIOWrapper(fv, encoding="utf-8"),
                io.TextIOWrapper(fz, encoding="utf-8"),
            ):
                vi = basic_clean(vi_line)
                zh = basic_clean(zh_line)
                if not vi or not zh:
                    continue
                if filter_cantonese and _looks_cantonese(zh):
                    continue
                if normalize_zh:
                    zh = _normalize_zh(zh)
                yield Pair(zh=zh, vi=vi)


def iter_pairs_from_jsonl(path: Path) -> Iterable[Pair]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield Pair(
                zh=basic_clean(d["zh"]),
                vi=basic_clean(d["vi"]),
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", default="configs/bi_mamba_55m.yaml")
    p.add_argument(
        "--preset",
        default="everyday",
        choices=list(PRESETS.keys()),
        help="Set of OPUS sources to download (ignored if --custom-jsonl is set).",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=f"Override --preset with an explicit list. Choose from: {sorted(SOURCES.keys())}",
    )
    p.add_argument("--max-train-pairs", type=int, default=None)
    p.add_argument("--max-valid-pairs", type=int, default=None)
    p.add_argument("--max-test-pairs", type=int, default=None)
    p.add_argument(
        "--custom-jsonl",
        default=None,
        help="Path to custom JSONL with {'zh': ..., 'vi': ...}. Skips OPUS download.",
    )
    p.add_argument("--out-dir", default=None, help="Override config.data.processed_dir.")
    p.add_argument("--cache-dir", default=None, help="Override config.data.raw_dir for downloads.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    out_dir = Path(args.out_dir or data_cfg["processed_dir"])
    cache_dir = Path(args.cache_dir or data_cfg.get("raw_dir", "data/raw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed = int(data_cfg.get("seed", 42))
    rng = random.Random(seed)

    min_len = int(data_cfg.get("min_len", 1))
    max_chars = int(data_cfg.get("max_len", 250)) * 4  # rough char limit
    script_check = bool(data_cfg.get("script_check", True))
    min_zh_vi_ratio = float(data_cfg.get("min_zh_vi_ratio", 0.10))
    max_zh_vi_ratio = float(data_cfg.get("max_zh_vi_ratio", 1.20))
    normalize_zh = bool(data_cfg.get("zh_normalize_simplified", True))
    filter_cantonese = bool(data_cfg.get("zh_filter_cantonese", True))
    if normalize_zh and _OPENCC_T2S is None:
        print(
            "  WARNING: zh_normalize_simplified=True but `opencc` is not installed; "
            "Traditional zh sentences will not be converted. Install with `pip install opencc`."
        )
        normalize_zh = False
    print(
        f"Filter: min_len={min_len} max_chars={max_chars} "
        f"zh/vi ratio=[{min_zh_vi_ratio:.2f}, {max_zh_vi_ratio:.2f}] "
        f"script_check={script_check} "
        f"zh_normalize_simplified={normalize_zh} zh_filter_cantonese={filter_cantonese}"
    )

    def _keep(zh: str, vi: str) -> bool:
        return pair_ok(
            zh,
            vi,
            min_len=min_len,
            max_chars=max_chars,
            min_zh_vi_ratio=min_zh_vi_ratio,
            max_zh_vi_ratio=max_zh_vi_ratio,
            script_check=script_check,
        )

    # Per-source cap so a single noisy huge source (e.g. OpenSubtitles, ~1.5M
    # pairs) cannot dominate the training mix. Pairs above the cap are
    # randomly subsampled before joining the global pool.
    raw_caps = data_cfg.get("max_pairs_per_source", {}) or {}
    caps: dict[str, int] = {k: int(v) for k, v in raw_caps.items() if v}

    pairs: List[Pair] = []
    seen: set[tuple[str, str]] = set()  # for dedup
    n_dropped = 0

    if args.custom_jsonl:
        print(f"Reading custom JSONL: {args.custom_jsonl}")
        for p_ in iter_pairs_from_jsonl(Path(args.custom_jsonl)):
            key = (p_.zh, p_.vi)
            if key in seen:
                continue
            if not _keep(p_.zh, p_.vi):
                n_dropped += 1
                continue
            seen.add(key)
            pairs.append(p_)
    else:
        source_keys = args.sources or PRESETS[args.preset]
        print(f"Preset: {args.preset}  sources: {source_keys}")
        if caps:
            print(f"Per-source caps: {caps}")
        for key in source_keys:
            if key not in SOURCES:
                print(f"  WARNING: unknown source '{key}', skipping")
                continue
            src = SOURCES[key]
            print(f"\n[{src.name}]")
            zip_path = cache_dir / f"{key}.zip"
            try:
                download(src.url, zip_path)
            except Exception as e:
                print(f"  ERROR: download failed ({e}); skipping {key}")
                continue
            cap = caps.get(key)
            n_src_dropped = 0
            src_pairs: List[Pair] = []
            for p_ in iter_pairs_from_opus_zip(
                zip_path,
                src,
                normalize_zh=normalize_zh,
                filter_cantonese=filter_cantonese,
            ):
                k = (p_.zh, p_.vi)
                if k in seen:
                    continue
                if not _keep(p_.zh, p_.vi):
                    n_src_dropped += 1
                    continue
                seen.add(k)
                src_pairs.append(p_)
            n_before_cap = len(src_pairs)
            if cap and len(src_pairs) > cap:
                rng.shuffle(src_pairs)
                src_pairs = src_pairs[:cap]
            pairs.extend(src_pairs)
            n_dropped += n_src_dropped
            cap_msg = (
                f"  (capped from {n_before_cap:,} -> {len(src_pairs):,})"
                if cap and n_before_cap > cap
                else ""
            )
            print(
                f"  + {len(src_pairs):,} new pairs{cap_msg}  "
                f"(dropped by filter: {n_src_dropped:,};  running total: {len(pairs):,})"
            )
    if n_dropped:
        print(f"\nTotal pairs dropped by filter: {n_dropped:,}")

    if not pairs:
        raise SystemExit(
            "No parallel pairs collected. Check internet / sources / "
            "or pass --custom-jsonl with your own data."
        )

    print(f"\nTotal unique, length-filtered pairs: {len(pairs):,}")
    rng.shuffle(pairs)

    n = len(pairs)
    n_test = int(args.max_test_pairs or data_cfg.get("max_test_pairs", 2000))
    n_valid = int(args.max_valid_pairs or data_cfg.get("max_valid_pairs", 2000))
    n_test = min(n_test, max(1, n // 50))
    n_valid = min(n_valid, max(1, n // 50))

    test_pairs = pairs[:n_test]
    valid_pairs = pairs[n_test : n_test + n_valid]
    train_pairs = pairs[n_test + n_valid :]

    cap_train = args.max_train_pairs or data_cfg.get("max_train_pairs")
    if cap_train is not None and len(train_pairs) > int(cap_train):
        train_pairs = train_pairs[: int(cap_train)]

    counts = {
        "train": write_jsonl(train_pairs, out_dir / "train.jsonl"),
        "valid": write_jsonl(valid_pairs, out_dir / "valid.jsonl"),
        "test":  write_jsonl(test_pairs,  out_dir / "test.jsonl"),
    }

    print("\nWrote:")
    for k, v in counts.items():
        print(f"  {out_dir/(k + '.jsonl')}: {v:,} pairs")


if __name__ == "__main__":
    main()
