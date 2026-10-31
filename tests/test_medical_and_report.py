import csv
import json
import numpy as np

from src.make_medical_set import generate, validate, MED_TERMS
from src.medical_metrics import summarize
from src.lid import pool_frames, collapse
from src import make_report


def test_generated_sentences_are_consistent():
    rows = generate(60)
    assert len(rows) == 60 and len({r["text"] for r in rows}) == 60
    assert validate(rows) == []
    assert "paracetamol" in MED_TERMS


def test_hand_written_template_is_valid():
    rows = list(csv.DictReader(open("data/medical/sentences_template.csv", encoding="utf-8")))
    assert len(rows) == 40 and validate(rows) == []


def test_medical_metrics():
    rows = [{"text": "Paracetamol 650 mg दिन में तीन बार", "drugs": "Paracetamol", "doses": "650 mg",
             "symptoms": "", "labs": ""},
            {"text": "आपका HbA1c 7 आया है", "drugs": "", "doses": "", "symptoms": "", "labs": "HbA1c 7"}]
    hyps = ["paracetamol 650mg दिन में तीन बार", "आपका एचबीए1सी 7 आया है"]
    s = summarize(rows, hyps)
    assert s["drug_recall"] == 1.0 and s["dose_acc"] == 1.0 and s["critical_exact"] == 1.0
    assert s["lab_recall"] == 0.0 and s["entity_recall"] == round(2 / 3, 4)
    wrong = summarize(rows[:1], ["paracetamol 500 mg"])
    assert wrong["dose_acc"] == 0.0 and wrong["critical_exact"] == 0.0


def test_pool_frames_ignores_padding():
    h = np.zeros((1, 1500, 4), dtype=np.float32)
    h[0, :50] = 1.0          # first second = real audio
    h[0, 50:] = 100.0        # padding region must be ignored
    out = pool_frames(h, [1.0])
    assert np.allclose(out, 1.0)
    assert collapse("hinglish") == "hi" and collapse("en") == "en"


def test_report_builds(tmp_path):
    rd = tmp_path
    def w(name, header, rows):
        with open(rd / name, "w", newline="", encoding="utf-8") as f:
            c = csv.writer(f); c.writerow(header); c.writerows(rows)
    h = ["model", "lang", "mode", "split", "n", "wer", "cer", "mer", "median_utt_wer", "halluc_rate"]
    w("baseline.csv", h, [["whisper-small", "hinglish", "forced", "test", 100, 2.1, 1.4, .9, 2.0, .52],
                          ["whisper-small", "hi", "auto", "test", 100, 1.2, .8, .8, 1.0, .16]])
    w("e2.csv", h, [["e2", "hinglish", "forced", "test", 100, .35, .2, .3, .3, .01]])
    (rd / "lid.json").write_text(json.dumps({"builtin_collapsed_acc": .9, "head_collapsed_acc": .97,
        "head_3class_acc": .85, "head_macro_f1": .84, "per_class_f1": {"hi": .8, "en": .95, "hinglish": .78},
        "confusion_labels": ["hi", "en", "hinglish"], "confusion": [[80, 1, 19], [0, 100, 0], [20, 0, 80]],
        "medical_probe_pred_share": {"hi": .1, "en": .0, "hinglish": .9}}))
    text = make_report.build(str(rd))
    assert "whisper-small (zero-shot)" in text and "**e2**" in text and "52%" in text
    assert "Confusion matrix" in text and "10% / 0% / 90%" in text
