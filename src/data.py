"""Phase 2: build a small, capped, 16 kHz training/eval manifest from public HF datasets.

Sources (all streamed, so only the rows we keep are downloaded):
  fleurs_hi : google/fleurs hi_in           (CC-BY-4.0, official speaker-disjoint splits)
  fleurs_en : google/fleurs en_us           (CC-BY-4.0, American English; NOT Indian-accented)
  hinglish  : addyo07/noisy-hinglish-asr    (Apache-2.0, code-switched; run --inspect first)

Usage:
  python -m src.data --inspect                      # look at columns/transcripts first
  python -m src.data --out /kaggle/working/data     # build the manifests
Output: <out>/audio/*.wav (16 kHz mono) + <out>/{train,validation,test}.jsonl
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
MIN_SEC, MAX_SEC = 1.0, 30.0  # Whisper context is 30 s
HINGLISH_REPO = "addyo07/noisy-hinglish-asr"


def to_16k(audio: dict) -> np.ndarray:
    """Decode an undecoded HF audio dict ({'bytes':..., 'path':...}) to 16 kHz mono float32."""
    data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data


def _probe(ds):
    next(iter(ds))  # force lazy errors (bad glob, script datasets) to surface here
    return ds


def load_fleurs(cfg: str, split: str):
    from datasets import load_dataset
    try:  # parquet route works on new `datasets` versions (no loading scripts)
        ds = load_dataset("parquet", split=split, streaming=True,
                          data_files={split: f"hf://datasets/google/fleurs/{cfg}/{split}/*.parquet"})
        return _probe(ds)
    except Exception as e:
        print(f"[fleurs {cfg}/{split}] parquet route failed ({type(e).__name__}); trying script route")
        ds = load_dataset("google/fleurs", cfg, split=split, streaming=True, trust_remote_code=True)
        return _probe(ds)


def _split_of(rel_parts):
    """Map a parquet path (relative to the repo root) to train/validation/test, or None."""
    if any("benchmark" in p.lower() for p in rel_parts):
        return None
    for p in reversed(rel_parts):
        p = p.lower()
        if p.startswith("train"):
            return "train"
        if p.startswith("val"):
            return "validation"
        if p.startswith("test"):
            return "test"
    return None


def hinglish_files():
    """Download all parquet shards once (cached) and group them by split."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(HINGLISH_REPO, repo_type="dataset", allow_patterns=["*.parquet"]))
    by_split = {"train": [], "validation": [], "test": []}
    for f in sorted(root.rglob("*.parquet")):
        sp = _split_of(f.relative_to(root).parts)
        if sp:
            by_split[sp].append(f)
    return by_split


def iter_hinglish(files, lang, clean_value, cap_per_file, seed=42):
    """Yield rows matching language==lang and clean condition. Works with either the
    'condition' or the 'noise_type' column name (the dataset card and live data differ)."""
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    for f in files:
        names = pq.read_schema(f).names
        cond = "condition" if "condition" in names else "noise_type"
        tbl = pq.read_table(f, filters=[("language", "=", lang), (cond, "=", clean_value)])
        if tbl.num_rows == 0:
            continue
        idx = rng.permutation(tbl.num_rows)[:cap_per_file]
        for row in tbl.take(idx).to_pylist():
            yield row


def dump(ds, name, split, n, out: Path, text_col, lang, fh, skip=0):
    """Write up to n clips from `ds` to disk and append records to manifest handle `fh`."""
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    kept = seen = 0
    for ex in ds:
        seen += 1
        if seen <= skip:
            continue
        text = (ex.get(text_col) or "").strip()
        if not text:
            continue
        wav = to_16k(ex["audio"])
        dur = len(wav) / SR
        if not (MIN_SEC <= dur <= MAX_SEC):
            continue
        rel = f"audio/{name}_{split}_{kept:06d}.wav"
        sf.write(out / rel, wav, SR, subtype="PCM_16")
        fh.write(json.dumps({"audio": rel, "text": text, "lang": lang, "source": name,
                             "split": split, "duration": round(dur, 2)}, ensure_ascii=False) + "\n")
        kept += 1
        if kept >= n:
            break
    print(f"{name}/{split}: kept {kept} clips")
    return kept


def inspect():
    """Show the REAL schema and value counts so we can set the filters correctly."""
    from collections import Counter
    import pyarrow.parquet as pq
    by_split = hinglish_files()
    print({k: len(v) for k, v in by_split.items()}, "parquet files per split")
    files = by_split["train"]
    print("columns:", pq.read_schema(files[0]).names)
    cnt = Counter()
    for f in files:
        names = pq.read_schema(f).names
        cols = [c for c in ("language", "condition", "noise_type", "source") if c in names]
        for r in pq.read_table(f, columns=cols).to_pylist():
            cnt[tuple((c, r[c]) for c in cols)] += 1
    print("--- value counts (train) ---")
    for k, v in cnt.most_common(40):
        print(v, dict(k))
    print("--- sample hinglish transcripts ---")
    for i, row in enumerate(iter_hinglish(files, "hinglish", "clean", 20)):
        print(repr(row["transcription"]))
        if i >= 9:
            break


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fhs = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "validation", "test")}
    try:
        # FLEURS: official splits are already speaker-disjoint
        for name, cfg, lang, n_tr in (("fleurs_hi", "hi_in", "hi", args.n_fleurs_hi),
                                      ("fleurs_en", "en_us", "en", args.n_fleurs_en)):
            for split, n in (("train", n_tr), ("validation", args.n_val), ("test", args.n_test)):
                dump(load_fleurs(cfg, split), name, split, n, out, "raw_transcription", lang, fhs[split])

        # Hinglish: use the dataset's own train/val/test shards (downloaded once, then filtered)
        by_split = hinglish_files()
        for split, n in (("train", args.n_hinglish), ("validation", args.n_val), ("test", args.n_test)):
            rows = iter_hinglish(by_split[split], args.hinglish_lang, args.clean_value,
                                 cap_per_file=max(50, n // 4))
            dump(rows, "hinglish", split, n, out, "transcription", "hinglish", fhs[split])
    finally:
        for f in fhs.values():
            f.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--n-fleurs-hi", type=int, default=2000)
    p.add_argument("--n-fleurs-en", type=int, default=1000)
    p.add_argument("--n-hinglish", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--hinglish-lang", default="hinglish")
    p.add_argument("--clean-value", default="clean")
    a = p.parse_args()
    inspect() if a.inspect else build(a)
