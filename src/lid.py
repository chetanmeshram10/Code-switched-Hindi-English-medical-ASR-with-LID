"""Utterance-level code-switch / language ID head on a FROZEN Whisper encoder.

Classes: hi | en | hinglish. Whisper's built-in LID can only say 'hi' or 'en' (no code-switch class), so we
compare on the collapsed task (hinglish -> hi) and report the 3-class head separately.

  python -m src.lid --data /kaggle/working/data
Outputs: models/lid_head.joblib, results/lid.json
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from .baseline import read_manifest

FRAMES_PER_SEC = 50  # Whisper encoder: 1500 frames per 30 s
CLASSES = ["hi", "en", "hinglish"]


def pool_frames(hidden, durations):
    """Mean-pool encoder frames over the real audio only (ignore the 30 s zero padding)."""
    out = []
    for h, d in zip(hidden, durations):
        n = max(1, min(h.shape[0], int(np.ceil(d * FRAMES_PER_SEC))))
        out.append(h[:n].mean(axis=0))
    return np.stack(out)


def collapse(label):
    return "en" if label == "en" else "hi"


def extract(model, proc, paths, durations, device, bs=16):
    import torch
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = proc.tokenizer
    sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
    lang_ids = [tok.convert_tokens_to_ids("<|hi|>"), tok.convert_tokens_to_ids("<|en|>")]
    embs, builtin = [], []
    for i in range(0, len(paths), bs):
        audios = [sf.read(p, dtype="float32")[0] for p in paths[i:i + bs]]
        feats = proc.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype)
        with torch.no_grad():
            enc = model.get_encoder()(feats)
            dec_in = torch.full((feats.shape[0], 1), sot, dtype=torch.long, device=device)
            logits = model(encoder_outputs=enc, decoder_input_ids=dec_in).logits[:, -1, :]
        pick = logits[:, lang_ids].argmax(-1).cpu().tolist()
        builtin += ["hi" if k == 0 else "en" for k in pick]
        embs.append(pool_frames(enc.last_hidden_state.float().cpu().numpy(), durations[i:i + bs]))
    return np.concatenate(embs), builtin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/kaggle/working/data")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--n-per-class", type=int, default=600)
    p.add_argument("--out-model", default="models/lid_head.joblib")
    p.add_argument("--results", default="results")
    p.add_argument("--medical-manifest", default="", help="optional: run the head on synthetic medical clips (all Hinglish)")
    a = p.parse_args()
    import joblib
    import torch
    import transformers
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    transformers.logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(a.model)
    model = WhisperForConditionalGeneration.from_pretrained(a.model).to(device)
    if device == "cuda":
        model = model.half()
    model.eval()

    rng = random.Random(0)
    train = []
    for c in CLASSES:
        rows = [r for r in read_manifest(a.data, "train") if r["lang"] == c]
        rng.shuffle(rows)
        train += rows[:a.n_per_class]
    test = read_manifest(a.data, "test")

    def feats(rows):
        return extract(model, proc, [str(Path(a.data) / r["audio"]) for r in rows], [r["duration"] for r in rows], device)

    Xtr, _ = feats(train)
    ytr = [r["lang"] for r in train]
    Xte, builtin = feats(test)
    yte = [r["lang"] for r in test]

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    clf.fit(Xtr, ytr)
    pred = list(clf.predict(Xte))
    res = {
        "n_train": len(train), "n_test": len(test),
        "builtin_collapsed_acc": round(accuracy_score([collapse(y) for y in yte], builtin), 4),
        "head_collapsed_acc": round(accuracy_score([collapse(y) for y in yte], [collapse(x) for x in pred]), 4),
        "head_3class_acc": round(accuracy_score(yte, pred), 4),
        "head_macro_f1": round(f1_score(yte, pred, average="macro", labels=CLASSES), 4),
        "per_class_f1": {c: round(v["f1-score"], 4) for c, v in
                         classification_report(yte, pred, labels=CLASSES, output_dict=True, zero_division=0).items()
                         if c in CLASSES},
        "confusion_labels": CLASSES,
        "confusion": confusion_matrix(yte, pred, labels=CLASSES).tolist(),
    }
    if a.medical_manifest and Path(a.medical_manifest).exists():
        mrows = [json.loads(l) for l in open(a.medical_manifest, encoding="utf-8")]
        root = Path(a.medical_manifest).parent
        Xm, _ = extract(model, proc, [str(root / r["audio"]) for r in mrows], [r["duration"] for r in mrows], device)
        from collections import Counter
        cnt = Counter(clf.predict(Xm))
        res["medical_probe_pred_share"] = {c: round(cnt.get(c, 0) / len(mrows), 4) for c in CLASSES}
    Path(a.results).mkdir(exist_ok=True)
    Path(a.results, "lid.json").write_text(json.dumps(res, indent=2))
    Path(a.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, a.out_model)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
