"""Print reference vs hypothesis side by side, worst utterances first.
Usage: python -m src.show_preds results/preds/whisper-small__forced__hinglish.jsonl -n 8
"""
import argparse
import json

import jiwer

from .normalize import normalize

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("-n", type=int, default=8)
p.add_argument("--worst", action="store_true", help="sort by per-utterance WER (default: first n)")
a = p.parse_args()

rows = [json.loads(l) for l in open(a.path, encoding="utf-8")]
for r in rows:
    rn = normalize(r["ref"])
    r["wer"] = jiwer.wer(rn, normalize(r["hyp"])) if rn else 0.0
if a.worst:
    rows.sort(key=lambda r: -r["wer"])
for r in rows[:a.n]:
    print(f"WER={r['wer']:.2f}  hyp_words={len(r['hyp'].split())}  ref_words={len(r['ref'].split())}")
    print("  REF:", r["ref"][:300])
    print("  HYP:", r["hyp"][:300])
