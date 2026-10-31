"""Evaluate systems on the synthetic medical probe set: WER/CER + entity recall + dose accuracy.

  python -m src.medical_eval --manifest /kaggle/working/medical/manifest.jsonl \
    --systems small=openai/whisper-small medium=openai/whisper-medium \
              e2=openai/whisper-small:/kaggle/working/checkpoints/e2/adapter
('name=base' for zero-shot, 'name=base:adapter_dir' for a LoRA adapter)
"""
import argparse
import csv
import json
from pathlib import Path

from .baseline import transcribe
from .medical_metrics import summarize
from .metrics import score, utt_stats


def load_system(spec, device):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    base, _, adapter = spec.partition(":")
    proc = WhisperProcessor.from_pretrained(base)
    model = WhisperForConditionalGeneration.from_pretrained(base)
    model.generation_config.forced_decoder_ids = None
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model = model.to(device)
    if device == "cuda":
        model = model.half()
    return proc, model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/kaggle/working/medical/manifest.jsonl")
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--results", default="results")
    p.add_argument("--batch-size", type=int, default=16)
    a = p.parse_args()
    import torch
    import transformers
    transformers.logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(a.manifest).parent
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8")]
    paths = [str(root / r["audio"]) for r in rows]
    durs = [r["duration"] for r in rows]
    refs = [r["text"] for r in rows]
    Path(a.results, "preds").mkdir(parents=True, exist_ok=True)
    table = []
    for item in a.systems:
        name, spec = item.split("=", 1)
        proc, model = load_system(spec, device)
        for mode in ("auto", "forced"):
            hyps = transcribe(model, proc, paths, durs, "hi" if mode == "forced" else None, a.batch_size, device)
            m = {**score(refs, hyps), **utt_stats(refs, hyps), **summarize(rows, hyps)}
            table.append({"model": name, "mode": mode, "n": len(rows),
                          **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}})
            print(table[-1])
            with open(Path(a.results, "preds", f"medical__{name}__{mode}.jsonl"), "w", encoding="utf-8") as f:
                for r, h in zip(rows, hyps):
                    f.write(json.dumps({"id": r["id"], "voice": r["voice"], "ref": r["text"], "hyp": h},
                                       ensure_ascii=False) + "\n")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(Path(a.results, "medical.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    print("saved", Path(a.results, "medical.csv"))


if __name__ == "__main__":
    main()
