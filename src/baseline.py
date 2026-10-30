"""Phase 3: zero-shot Whisper baseline (the 'before' in the before/after table).

For each model x language set x decoding mode we report WER / CER / MER with the shared normalizer.
  mode 'auto'   : language token NOT given (Whisper detects the language itself)
  mode 'forced' : language token forced (hi -> hi, en -> en, hinglish -> hi)

Usage (GPU):
  python -m src.baseline --data /kaggle/working/data --models openai/whisper-small openai/whisper-medium
Outputs: results/baseline.csv and results/preds/<model>__<mode>__<lang>.jsonl (hyp vs ref, for error analysis)
"""
import argparse
import csv
import json
from pathlib import Path

import soundfile as sf

from .metrics import score, utt_stats

FORCED = {"hi": "hi", "en": "en", "hinglish": "hi"}
LANGS = ("hi", "en", "hinglish")


def read_manifest(data_dir, split):
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def transcribe(model, proc, paths, durations, language, batch_size, device):
    """Greedy decoding. Clips are sorted by duration and the token budget scales with clip length
    (a 3 s clip cannot legitimately need 400 tokens), which stops runaway repetition loops."""
    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32
    order = sorted(range(len(paths)), key=lambda i: durations[i])
    hyps = [None] * len(paths)
    for s0 in range(0, len(order), batch_size):
        idx = order[s0:s0 + batch_size]
        audios = [sf.read(paths[i], dtype="float32")[0] for i in idx]
        feats = proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        max_tokens = min(440, int(max(durations[i] for i in idx) * 20) + 30)
        kwargs = {"task": "transcribe", "max_new_tokens": max_tokens}
        if language:
            kwargs["language"] = language
        with torch.no_grad():
            ids = model.generate(feats.to(device, dtype), **kwargs)
        out = proc.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, h in zip(idx, out):
            hyps[i] = h.strip()
    return hyps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/kaggle/working/data")
    p.add_argument("--models", nargs="+", default=["openai/whisper-small"])
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--results", default="results")
    a = p.parse_args()

    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    transformers.logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    rows = read_manifest(a.data, a.split)
    res_dir = Path(a.results)
    (res_dir / "preds").mkdir(parents=True, exist_ok=True)
    table = []
    for name in a.models:
        proc = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        model.generation_config.forced_decoder_ids = None
        for lang in LANGS:
            subset = [r for r in rows if r["lang"] == lang]
            paths = [str(Path(a.data) / r["audio"]) for r in subset]
            refs = [r["text"] for r in subset]
            durs = [r["duration"] for r in subset]
            for mode in ("auto", "forced"):
                hyps = transcribe(model, proc, paths, durs, FORCED[lang] if mode == "forced" else None,
                                  a.batch_size, device)
                m = score(refs, hyps)
                table.append({"model": name.split("/")[-1], "lang": lang, "mode": mode, "split": a.split,
                              "n": len(subset), **{k: round(v, 4) for k, v in {**m, **utt_stats(refs, hyps)}.items()}})
                print(table[-1])
                out = res_dir / "preds" / f"{name.split('/')[-1]}__{mode}__{lang}.jsonl"
                with open(out, "w", encoding="utf-8") as f:
                    for r, h in zip(subset, hyps):
                        f.write(json.dumps({"audio": r["audio"], "ref": r["text"], "hyp": h},
                                           ensure_ascii=False) + "\n")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(res_dir / "baseline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print("saved", res_dir / "baseline.csv")


if __name__ == "__main__":
    main()
