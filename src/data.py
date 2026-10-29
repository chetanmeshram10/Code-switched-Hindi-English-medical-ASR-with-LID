"""Phase 2: build a small, capped, 16 kHz training/eval manifest from public HF datasets.

Sources (all streamed, so only the rows we keep are downloaded):
  fleurs_hi : google/fleurs hi_in           (CC-BY-4.0, official speaker-disjoint splits)
  fleurs_en : google/fleurs en_us           (CC-BY-4.0, American English; NOT Indian-accented)
  hinglish  : addyo07/noisy-hinglish-asr    (Apache-2.0, code-switched; verify script with --inspect)

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


def load_hinglish(split: str = "train"):
    from datasets import Audio, load_dataset
    ds = load_dataset(HINGLISH_REPO, split=split, streaming=True)
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    return ds


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
    """Print columns + a few transcripts so you can confirm the script (Devanagari vs Latin)."""
    ds = load_hinglish("train")
    for i, ex in enumerate(ds):
        row = {k: (v if k != "audio" else "<audio>") for k, v in ex.items()}
        print(row)
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

        # Hinglish: single stream -> disjoint row ranges for train / val / test
        def hinglish_stream():
            ds = load_hinglish("train")
            ds = ds.filter(lambda e: e.get("language") == "hinglish" and e.get("noise_type") == "clean")
            return ds.shuffle(seed=42, buffer_size=2000)

        a, b = args.n_hinglish, args.n_hinglish + args.n_val
        dump(hinglish_stream(), "hinglish", "train", a, out, "transcription", "hinglish", fhs["train"])
        dump(hinglish_stream(), "hinglish", "validation", args.n_val, out, "transcription", "hinglish",
             fhs["validation"], skip=a)
        dump(hinglish_stream(), "hinglish", "test", args.n_test, out, "transcription", "hinglish",
             fhs["test"], skip=b)
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
    a = p.parse_args()
    inspect() if a.inspect else build(a)
