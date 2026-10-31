"""Why does the model miss drug names and doses? Categorises misses from a medical preds file.

  python -m src.medical_errors results/preds/medical__e2__forced.jsonl --manifest /kaggle/working/medical/manifest.jsonl
Writes results/medical_errors.md (summary + examples) and prints it.

Drug miss categories (heuristic, string similarity on normalized text):
  latin_misspelling : a Latin-script word in the output is close to the drug name (similarity >= 0.75)
  other_or_devanagari: no close Latin word (drug was dropped, replaced, or written in Devanagari)
Dose miss categories:
  number_ok_unit_differs : the number is present but 'mg'/'mcg'/... is written differently or missing
  number_wrong_or_missing: the number itself is wrong or absent
"""
import argparse
import difflib
import json
from collections import Counter
from pathlib import Path

from .medical_metrics import prep
from .normalize import normalize, word_script


def _latin_candidates(hyp):
    toks = [t for t in normalize(hyp).split() if word_script(t) == "en"]
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


def classify_drug(drug, hyp):
    d = normalize(drug)
    best, tok = 0.0, ""
    for c in _latin_candidates(hyp):
        r = difflib.SequenceMatcher(None, d, c).ratio()
        if r > best:
            best, tok = r, c
    return ("latin_misspelling", tok) if best >= 0.75 else ("other_or_devanagari", tok)


def classify_dose(dose, hyp):
    num = dose.split()[0]
    return "number_ok_unit_differs" if f" {num} " in prep(hyp) else "number_wrong_or_missing"


def analyse(rows, hyps):
    drug_c, dose_c, examples = Counter(), Counter(), []
    n_drug = n_dose = 0
    for r, h in zip(rows, hyps):
        hp = prep(h)
        for d in filter(None, r.get("drugs", "").split(";")):
            n_drug += 1
            if prep(d) not in hp:
                cat, tok = classify_drug(d, h)
                drug_c[cat] += 1
                examples.append((f"drug '{d}' [{cat}]", r["text"], h))
        for x in filter(None, r.get("doses", "").split(";")):
            n_dose += 1
            if prep(x) not in hp:
                cat = classify_dose(x, h)
                dose_c[cat] += 1
                examples.append((f"dose '{x}' [{cat}]", r["text"], h))
    return n_drug, n_dose, drug_c, dose_c, examples


def render(n_drug, n_dose, drug_c, dose_c, examples, k=14):
    lines = [f"### Drug-name misses ({sum(drug_c.values())} of {n_drug})", ""]
    lines += [f"- {c}: {n}" for c, n in drug_c.most_common()]
    lines += ["", f"### Dose misses ({sum(dose_c.values())} of {n_dose})", ""]
    lines += [f"- {c}: {n}" for c, n in dose_c.most_common()]
    lines += ["", "### Examples", ""]
    for tag, ref, hyp in examples[:k]:
        lines += [f"- **{tag}**", f"  - REF: {ref}", f"  - HYP: {hyp}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("preds")
    p.add_argument("--manifest", default="/kaggle/working/medical/manifest.jsonl")
    p.add_argument("--out", default="results/medical_errors.md")
    a = p.parse_args()
    meta = {(r["id"], r["voice"]): r for r in map(json.loads, open(a.manifest, encoding="utf-8"))}
    preds = [json.loads(l) for l in open(a.preds, encoding="utf-8")]
    rows = [meta[(p_["id"], p_["voice"])] for p_ in preds]
    text = render(*analyse(rows, [p_["hyp"] for p_ in preds]))
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)
