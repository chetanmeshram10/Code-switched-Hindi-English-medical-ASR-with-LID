"""Phase 2: build a small, capped, 16 kHz manifest from addyo07/noisy-hinglish-asr.

Verified from --inspect on the live dataset:
  columns : transcription, audio, language, condition, source, speaker_id, ...
  hi       -> indicvoices_train / fleurs_hi_in_train   (Devanagari)
  en       -> ami_ihm_train / fleurs_en_us_train       (meeting + read English)
  hinglish -> mucs_train                               (Devanagari + Latin, our target format)
  condition in {clean, synthetic_noise}; we use ONLY 'clean' rows (noisy copies are for Phase 6+).

Splits: train from the 'train' shards; validation/test are carved from the held-out shards
BY SPEAKER (train speakers are excluded), so there is no speaker leakage.

Usage:
  python -m src.data --inspect
  python -m src.data --out /kaggle/working/data
Output: <out>/audio/*.wav (16 kHz mono), <out>/{train,validation,test}.jsonl, <out>/split_report.json
"""
import argparse
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
MIN_SEC, MAX_SEC = 1.0, 30.0  # Whisper context is 30 s
REPO = "addyo07/noisy-hinglish-asr"
# MUCS label noise: Devanagari glued to Latin with no space, e.g. "अरैarray" -> drop these clips
_GLUED = re.compile(r"[\u0900-\u097F][A-Za-z]|[A-Za-z][\u0900-\u097F]")


def to_16k(audio: dict) -> np.ndarray:
    """Decode an HF audio dict ({'bytes':..., 'path':...}) to 16 kHz mono float32."""
    data, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    return data


def _split_of(rel_parts):
    """Map a parquet path (relative to repo root) to train/validation/test, or None."""
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


def dataset_files():
    """Download all parquet shards once (cached) and group by split. 'other' = unclassified/benchmark."""
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=["*.parquet"]))
    by = {"train": [], "validation": [], "test": [], "other": []}
    for f in sorted(root.rglob("*.parquet")):
        by[_split_of(f.relative_to(root).parts) or "other"].append(f)
    return by, root


def _cond_col(names):
    return "condition" if "condition" in names else "noise_type"


def iter_rows(files, lang, clean_value, cap_per_file, source_substr=None, speakers=None, seed=42):
    """Yield rows with language==lang and clean condition (optionally filtered by source/speaker),
    sampling at most cap_per_file random rows per shard, shards in random order.

    Memory-safe: metadata is read WITHOUT the audio column, then audio is streamed in small
    batches and only the selected rows are materialised (a plain table.take() on big English
    shards overflows Arrow's 2 GB binary-offset limit)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    rng = np.random.default_rng(seed)
    files = list(files)
    rng.shuffle(files)
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        cond = _cond_col(names)
        meta_cols = [c for c in ("language", cond, "source", "speaker_id") if c in names]
        meta = pf.read(columns=meta_cols).to_pydict()
        n = len(meta["language"])
        src = meta.get("source", [None] * n)
        spk = meta.get("speaker_id", [None] * n)
        keep = [i for i in range(n)
                if meta["language"][i] == lang and meta[cond][i] == clean_value
                and (not source_substr or source_substr in (src[i] or ""))
                and (speakers is None or spk[i] in speakers)]
        rng.shuffle(keep)
        keep = sorted(keep[:cap_per_file])
        if not keep:
            continue
        want = set(keep)
        offset = 0
        for batch in pf.iter_batches(batch_size=16):
            m = batch.num_rows
            local = [i - offset for i in range(offset, offset + m) if i in want]
            if local:
                for row in batch.take(pa.array(local)).to_pylist():
                    yield row
            offset += m
            if offset > keep[-1]:
                break


def speaker_set(files, lang, clean_value, source_substr=None):
    """Distinct non-null speaker ids for a language (metadata columns only, no audio read)."""
    import pyarrow.parquet as pq
    out = set()
    for f in files:
        names = pq.read_schema(f).names
        if "speaker_id" not in names:
            continue
        cond = _cond_col(names)
        cols = [c for c in ("language", cond, "source", "speaker_id") if c in names]
        tbl = pq.read_table(f, columns=cols, filters=[("language", "=", lang), (cond, "=", clean_value)])
        spk = tbl.column("speaker_id").to_pylist()
        src = tbl.column("source").to_pylist() if "source" in cols else [None] * len(spk)
        out |= {s for s, so in zip(spk, src) if s and (not source_substr or source_substr in (so or ""))}
    return out


def split_heldout(held_files, lang, clean_value, source_substr=None, exclude=frozenset()):
    """Split held-out shards into validation/test with DISJOINT speakers (and disjoint from `exclude`).
    Returns {'validation': (files, speakers), 'test': (files, speakers)}."""
    spk = sorted(speaker_set(held_files, lang, clean_value, source_substr) - set(exclude))
    if len(spk) >= 2:
        return {"validation": (held_files, set(spk[::2])), "test": (held_files, set(spk[1::2]))}
    print(f"[warn] <2 usable held-out speakers for '{lang}'; splitting by shard instead (leakage not guaranteed)")
    fs = sorted(held_files)
    half = max(1, len(fs) // 2)
    return {"validation": (fs[:half], None), "test": (fs[half:] or fs[:half], None)}


def dump(rows, name, split, n, out: Path, text_col, lang, fh, text_ok=None):
    """Write up to n clips to disk and append records to manifest handle `fh`."""
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    kept = dropped = 0
    for ex in rows:
        text = (ex.get(text_col) or "").strip()
        if not text:
            continue
        if text_ok and not text_ok(text):
            dropped += 1
            continue
        wav = to_16k(ex["audio"])
        dur = len(wav) / SR
        if not (MIN_SEC <= dur <= MAX_SEC):
            continue
        rel = f"audio/{name}_{split}_{kept:06d}.wav"
        sf.write(out / rel, wav, SR, subtype="PCM_16")
        fh.write(json.dumps({"audio": rel, "text": text, "lang": lang, "source": ex.get("source"),
                             "speaker": ex.get("speaker_id"), "split": split,
                             "duration": round(dur, 2)}, ensure_ascii=False) + "\n")
        kept += 1
        if kept >= n:
            break
    print(f"{name}/{split}: kept {kept} clips" + (f" (dropped {dropped} glued-script labels)" if dropped else ""))
    return kept


def inspect():
    """Show shard layout, value counts per split group, and sample Hinglish transcripts."""
    import pyarrow.parquet as pq
    by, root = dataset_files()
    print({k: len(v) for k, v in by.items()}, "parquet files per split")
    print("unclassified/benchmark files:", [str(f.relative_to(root)) for f in by["other"]][:12])
    print("columns:", pq.read_schema(by["train"][0]).names)
    for group, files in (("train", by["train"]), ("held-out", by["validation"] + by["test"])):
        cnt = Counter()
        for f in files:
            names = pq.read_schema(f).names
            cols = [c for c in ("language", "condition", "noise_type", "source") if c in names]
            for r in pq.read_table(f, columns=cols).to_pylist():
                cnt[tuple((c, r[c]) for c in cols)] += 1
        print(f"--- value counts ({group}) ---")
        for k, v in cnt.most_common(30):
            print(v, dict(k))
    for lang in ("hi", "en", "hinglish"):
        tr = speaker_set(by["train"], lang, "clean")
        ho = speaker_set(by["validation"] + by["test"], lang, "clean")
        print(f"speakers[{lang}]: train={len(tr)} held-out={len(ho)} overlap={len(tr & ho)}")
    print("--- sample hinglish transcripts ---")
    for i, row in enumerate(iter_rows(by["train"], "hinglish", "clean", 20)):
        print(repr(row["transcription"]))
        if i >= 9:
            break


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    by, _ = dataset_files()
    held = by["validation"] + by["test"]
    if not held:
        raise SystemExit("No validation/test shards found; run --inspect and check the unclassified paths.")
    clean = args.clean_value
    fhs = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "validation", "test")}
    report = {}
    try:
        for lang, n_tr in (("hi", args.n_hi), ("en", args.n_en), ("hinglish", args.n_hinglish)):
            sub = args.en_source if lang == "en" else None
            train_spk = speaker_set(by["train"], lang, clean)
            pools = split_heldout(held, lang, clean, sub, exclude=train_spk)
            report[lang] = {"train_speakers": len(train_spk),
                            "val_speakers": len(pools["validation"][1] or []),
                            "test_speakers": len(pools["test"][1] or [])}
            ok = (lambda t: not _GLUED.search(t)) if lang == "hinglish" else None
            plan = (("train", by["train"], None, n_tr),
                    ("validation", *pools["validation"], args.n_val),
                    ("test", *pools["test"], args.n_test))
            for split, files, spks, n in plan:
                rows = iter_rows(files, lang, clean, max(100, n // 2), sub, spks)
                dump(rows, lang, split, n, out, "transcription", lang, fhs[split], text_ok=ok)
    finally:
        for f in fhs.values():
            f.close()
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    print("split report:", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--out", default="data/processed")
    p.add_argument("--n-hi", type=int, default=2000)
    p.add_argument("--n-en", type=int, default=1000)
    p.add_argument("--n-hinglish", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=200)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--clean-value", default="clean")
    p.add_argument("--en-source", default=None, help="e.g. 'fleurs' to use only FLEURS English (train only has ~1.5k)")
    a = p.parse_args()
    inspect() if a.inspect else build(a)
