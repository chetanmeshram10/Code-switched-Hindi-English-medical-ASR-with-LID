"""Phase 4: LoRA fine-tuning of Whisper-small on Hindi / English / Hinglish.

Experiments (the ablation table):
  e1 : --langs hi en                  (no code-switched data)
  e2 : --langs hi en hinglish         (+ real code-switched speech)
  e3 : --langs hi en hinglish --augment   (+ speed/noise/SpecAugment)

Usage (GPU):
  python -m src.train --name e2 --langs hi en hinglish --epochs 3
Outputs: checkpoints/<name>/ (LoRA adapter), results/<name>.csv (WER/CER/MER per lang x mode x split),
         results/<name>_config.json
"""
import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# One GPU only: on Kaggle "T4 x2" the Trainer would wrap the model in nn.DataParallel, which gathers the
# vocab-sized logits on GPU 0 and runs out of memory. Must be set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .baseline import FORCED, read_manifest, transcribe
from .metrics import score, utt_stats

MAX_LABEL_TOKENS = 440  # Whisper decoder limit is 448


def augment_wave(wav: np.ndarray, rng) -> np.ndarray:
    """Waveform augmentation: speed perturbation (0.9/1.0/1.1) and additive noise at 5-25 dB SNR."""
    if rng.random() < 0.5:
        f = float(rng.choice([0.9, 1.1]))
        n = max(1, int(len(wav) / f))
        wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav)
    if rng.random() < 0.6:
        snr_db = rng.uniform(5, 25)
        power = float(np.mean(wav ** 2)) + 1e-10
        wav = wav + rng.normal(0.0, np.sqrt(power / 10 ** (snr_db / 10)), len(wav))
    return wav.astype(np.float32)


class ManifestDataset:
    """Computes log-mel features on the fly (precomputing 8k x 1 MB would not fit in RAM)."""

    def __init__(self, rows, data_dir, processor, augment=False):
        self.data_dir, self.proc, self.augment = Path(data_dir), processor, augment
        self.rng = np.random.default_rng()
        tok = processor.tokenizer
        self.rows, self.labels = [], []
        for r in rows:
            tok.set_prefix_tokens(language=FORCED[r["lang"]], task="transcribe")
            ids = tok(r["text"]).input_ids  # <|sot|><|lang|><|transcribe|><|notimestamps|> text <|eot|>
            if len(ids) <= MAX_LABEL_TOKENS:
                self.rows.append(r)
                self.labels.append(ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        wav, _ = sf.read(self.data_dir / self.rows[i]["audio"], dtype="float32")
        if self.augment:
            wav = augment_wave(wav, self.rng)
        feats = self.proc.feature_extractor(wav, sampling_rate=16000).input_features[0]
        return {"input_features": feats, "labels": self.labels[i]}


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        import torch
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")
        lab = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]  # the model prepends the start token itself when shifting labels
        batch["labels"] = labels
        return batch


def evaluate(model, processor, data_dir, splits, out_csv, name, batch_size=16):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.merge_and_unload().to(device)
    if device == "cuda":
        model = model.half()
    model.eval()
    table = []
    for split in splits:
        rows = read_manifest(data_dir, split)
        for lang in ("hi", "en", "hinglish"):
            subset = [r for r in rows if r["lang"] == lang]
            if not subset:
                continue
            paths = [str(Path(data_dir) / r["audio"]) for r in subset]
            refs, durs = [r["text"] for r in subset], [r["duration"] for r in subset]
            for mode in ("auto", "forced"):
                hyps = transcribe(model, processor, paths, durs, FORCED[lang] if mode == "forced" else None,
                                  batch_size, device)
                m = {**score(refs, hyps), **utt_stats(refs, hyps)}
                table.append({"model": name, "lang": lang, "mode": mode, "split": split, "n": len(subset),
                              **{k: round(v, 4) for k, v in m.items()}})
                print(table[-1])
                preds = Path(out_csv).parent / "preds" / f"{name}__{split}__{mode}__{lang}.jsonl"
                preds.parent.mkdir(parents=True, exist_ok=True)
                with open(preds, "w", encoding="utf-8") as f:
                    for r, h in zip(subset, hyps):
                        f.write(json.dumps({"audio": r["audio"], "ref": r["text"], "hyp": h},
                                           ensure_ascii=False) + "\n")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--data", default="/kaggle/working/data")
    p.add_argument("--base", default="openai/whisper-small")
    p.add_argument("--langs", nargs="+", default=["hi", "en", "hinglish"])
    p.add_argument("--augment", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--eval-splits", nargs="+", default=["validation", "test"])
    p.add_argument("--ckpt-dir", default="/kaggle/working/checkpoints")
    p.add_argument("--results", default="results")
    a = p.parse_args()

    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor)
    transformers.logging.set_verbosity_error()

    processor = WhisperProcessor.from_pretrained(a.base)
    model = WhisperForConditionalGeneration.from_pretrained(a.base)  # fp32 weights; Trainer uses fp16 autocast
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    if a.augment:  # SpecAugment inside the encoder (only active in training mode)
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = 0.05
        model.config.mask_feature_length = 10
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]))
    model.print_trainable_parameters()

    train_rows = [r for r in read_manifest(a.data, "train") if r["lang"] in a.langs]
    val_rows = [r for r in read_manifest(a.data, "validation") if r["lang"] in a.langs]
    train_ds = ManifestDataset(train_rows, a.data, processor, augment=a.augment)
    val_ds = ManifestDataset(val_rows, a.data, processor)
    hours = sum(r["duration"] for r in train_rows) / 3600
    print(f"train clips: {len(train_ds)} ({hours:.2f} h) | val clips: {len(val_ds)}")

    out_dir = Path(a.ckpt_dir) / a.name
    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, warmup_steps=100, num_train_epochs=a.epochs, lr_scheduler_type="linear",
        fp16=torch.cuda.is_available(), eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        logging_steps=25, report_to="none", remove_unused_columns=False, label_names=["labels"],
        dataloader_num_workers=2, seed=42)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                             data_collator=Collator(processor, model.config.decoder_start_token_id))
    trainer.train()
    model.save_pretrained(str(out_dir / "adapter"))

    Path(a.results).mkdir(exist_ok=True)
    Path(a.results, f"{a.name}_config.json").write_text(json.dumps(
        {**vars(a), "train_clips": len(train_ds), "train_hours": round(hours, 2)}, indent=2))
    evaluate(model, processor, a.data, a.eval_splits, Path(a.results) / f"{a.name}.csv", a.name)


if __name__ == "__main__":
    main()
