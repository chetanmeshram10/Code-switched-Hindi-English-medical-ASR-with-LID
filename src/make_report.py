"""Turn results/*.csv|json into markdown tables and inject them into README.md between
<!-- RESULTS:START --> and <!-- RESULTS:END -->. Also writes results/RESULTS.md.

  python -m src.make_report
"""
import csv
import glob
import json
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _csv(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v, pct=False):
    if v in (None, "", "None"):
        return "-"
    x = float(v)
    return f"{x * 100:.0f}%" if pct else f"{x:.2f}"


def _pick(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def _md(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def asr_tables(rd):
    systems = []
    base = _csv(Path(rd, "baseline.csv"))
    for name in ("whisper-small", "whisper-medium"):
        rows = [r for r in base if r["model"] == name]
        if rows:
            systems.append((f"{name} (zero-shot)", rows))
    for f in sorted(glob.glob(str(Path(rd, "e[0-9]*.csv")))):
        stem = Path(f).stem
        systems.append((f"**{stem}** (LoRA whisper-small)", _csv(f)))
    main_rows, mode_rows = [], []
    for label, rows in systems:
        g = lambda lang, mode, key: (_pick(rows, lang=lang, mode=mode, split="test") or {}).get(key)
        main_rows.append([label, _f(g("hi", "forced", "wer")), _f(g("en", "forced", "wer")),
                          _f(g("hinglish", "forced", "wer")), _f(g("hinglish", "forced", "cer")),
                          _f(g("hinglish", "forced", "halluc_rate"), pct=True)])
        mode_rows.append([label] + [f"{_f(g(l, 'auto', 'wer'))} / {_f(g(l, 'forced', 'wer'))}" for l in ("hi", "en", "hinglish")])
    t1 = _md(["System", "Hindi WER", "English WER", "Hinglish WER", "Hinglish CER", "Hinglish hallucination rate"], main_rows)
    t2 = _md(["System", "Hindi (auto / forced)", "English (auto / forced)", "Hinglish (auto / forced)"], mode_rows)
    return t1, t2


def medical_table(rd):
    rows = [r for r in _csv(Path(rd, "medical.csv")) if r["mode"] == "forced"]
    if not rows:
        return None
    body = [[r["model"], _f(r["wer"]), _f(r["drug_recall"], True), _f(r["dose_acc"], True),
             _f(r["symptom_recall"], True), _f(r["lab_recall"], True), _f(r["critical_exact"], True)] for r in rows]
    return _md(["System", "WER", "Drug recall", "Dose accuracy", "Symptom recall", "Lab recall", "Critical-exact"], body)


def lid_table(rd):
    p = Path(rd, "lid.json")
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    probe = r.get("medical_probe_pred_share")
    extra = [["Head on synthetic medical clips (all Hinglish): predicted hi / en / hinglish",
              " / ".join(_f(probe[c], True) for c in ("hi", "en", "hinglish"))]] if probe else []
    t = _md(["Metric", "Value"],
            [["Whisper built-in LID, hi/en collapsed accuracy", _f(r["builtin_collapsed_acc"], True)],
             ["Our head, hi/en collapsed accuracy", _f(r["head_collapsed_acc"], True)],
             ["Our head, 3-class accuracy (hi / en / hinglish)", _f(r["head_3class_acc"], True)],
             ["Our head, macro-F1", _f(r["head_macro_f1"])]] +
            [[f"F1 ({c})", _f(v)] for c, v in r["per_class_f1"].items()] + extra)
    cm = _md(["gold \\ predicted"] + r["confusion_labels"],
             [[lab] + [str(x) for x in row] for lab, row in zip(r["confusion_labels"], r["confusion"])])
    return t + "\n\nConfusion matrix (test set):\n\n" + cm


def build(rd="results"):
    t1, t2 = asr_tables(rd)
    parts = ["### ASR on held-out public test sets (test split, forced language)\n\n" + t1,
             "### Does language detection matter? (WER, auto vs forced language token)\n\n" + t2]
    m, l = medical_table(rd), lid_table(rd)
    if m:
        parts.append("### Medical probe set (synthetic speech, forced Hindi decoding)\n\n" + m)
    if l:
        parts.append("### Utterance-level language ID\n\n" + l)
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    text = build()
    Path("results").mkdir(exist_ok=True)
    Path("results/RESULTS.md").write_text(text, encoding="utf-8")
    readme = Path("README.md")
    if readme.exists():
        s = readme.read_text(encoding="utf-8")
        if START in s and END in s:
            s = s[:s.index(START) + len(START)] + "\n" + text + s[s.index(END):]
            readme.write_text(s, encoding="utf-8")
            print("README.md updated")
    print(text)
