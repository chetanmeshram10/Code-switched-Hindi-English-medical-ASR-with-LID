"""Build the medical probe-set sentences: 40 hand-written (data/medical/sentences_template.csv)
+ N template-generated Hinglish sentences with entity annotations (drugs, doses, symptoms, labs).
Output: data/medical/medical_set.csv   (committed; audio is generated separately by tts_medical.py)
"""
import argparse
import csv
import random
from pathlib import Path

DRUGS = [("Paracetamol", ["500 mg", "650 mg"]), ("Metformin", ["500 mg", "1000 mg"]),
         ("Amoxicillin", ["250 mg", "500 mg"]), ("Azithromycin", ["250 mg", "500 mg"]),
         ("Pantoprazole", ["20 mg", "40 mg"]), ("Cetirizine", ["5 mg", "10 mg"]),
         ("Ibuprofen", ["200 mg", "400 mg"]), ("Atorvastatin", ["10 mg", "20 mg", "40 mg"]),
         ("Amlodipine", ["2 mg", "5 mg"]), ("Levothyroxine", ["25 mcg", "50 mcg", "100 mcg"]),
         ("Omeprazole", ["20 mg"]), ("Ciprofloxacin", ["250 mg", "500 mg"]), ("Aspirin", ["75 mg"]),
         ("Prednisolone", ["10 mg", "20 mg"]), ("Losartan", ["25 mg", "50 mg"]),
         ("Telmisartan", ["40 mg"]), ("Clopidogrel", ["75 mg"]), ("Ondansetron", ["4 mg"]),
         ("Domperidone", ["10 mg"]), ("Montelukast", ["10 mg"]), ("Doxycycline", ["100 mg"]),
         ("Diclofenac", ["50 mg"]), ("Metoprolol", ["25 mg", "50 mg"]), ("Glimepiride", ["1 mg", "2 mg"]),
         ("Cefixime", ["200 mg"]), ("Rabeprazole", ["20 mg"]), ("Insulin", ["10 units", "12 units"]),
         ("Vitamin D", ["60000 units"]), ("Dolo", ["650 mg"])]
FREQ = ["सुबह खाने के बाद", "रात को सोने से पहले", "दिन में दो बार", "दिन में तीन बार",
        "सुबह खाली पेट", "हर आठ घंटे में", "दर्द होने पर", "रोज़ एक बार"]
DUR = ["तीन दिन तक", "पाँच दिन तक", "एक हफ़्ते तक", "दस दिन तक", "", ""]
SYMPTOMS = ["बुखार", "सिर दर्द", "खाँसी", "गले में खराश", "पेट में दर्द", "उल्टी", "दस्त", "चक्कर",
            "कमज़ोरी", "जोड़ों में दर्द", "सीने में दर्द", "साँस लेने में तकलीफ़", "पीठ दर्द", "थकान",
            "नींद न आना", "भूख न लगना"]
LABS = [("blood pressure", ["120 by 80", "140 by 90", "150 by 95"]), ("sugar fasting", ["96", "126", "148"]),
        ("haemoglobin", ["9", "11", "13"]), ("creatinine", ["1", "2"]), ("TSH", ["3", "6", "8"]),
        ("HbA1c", ["6", "7", "9"]), ("cholesterol", ["180", "220", "240"]), ("platelet count", ["90000", "150000"])]
MED_TERMS = sorted({w.lower() for name, _ in DRUGS for w in name.split()})


def _join(*parts):
    return " ".join(p for p in parts if p)


def generate(n, seed=7):
    rng = random.Random(seed)
    rows, seen = [], set()
    kinds = ["drug"] * 4 + ["combo"] * 2 + ["symptom"] * 2 + ["lab"] * 2
    while len(rows) < n:
        kind = rng.choice(kinds)
        drugs = doses = symptoms = labs = []
        if kind == "drug":
            d, ds = rng.choice(DRUGS)
            dose = rng.choice(ds)
            tpl = rng.choice(["{d} {x} {f} {u} लीजिए", "आपको {d} {x} {f} {u} लेनी है", "मरीज़ को {d} {x} {f} {u} दीजिए"])
            text = _join(*tpl.format(d=d, x=dose, f=rng.choice(FREQ), u=rng.choice(DUR)).split())
            drugs, doses = [d], [dose]
        elif kind == "combo":
            d, ds = rng.choice(DRUGS)
            dose, s = rng.choice(ds), rng.choice(SYMPTOMS)
            text = _join(s, "के लिए", d, dose, rng.choice(FREQ), "लीजिए")
            drugs, doses, symptoms = [d], [dose], [s]
        elif kind == "symptom":
            s1, s2 = rng.sample(SYMPTOMS, 2)
            text = rng.choice(["मरीज़ को {a} और {b} की शिकायत है", "{a} और {b} कब से है"]).format(a=s1, b=s2)
            symptoms = [s1, s2]
        else:
            lab, vals = rng.choice(LABS)
            v = rng.choice(vals)
            text = rng.choice(["आपका {l} {v} आया है", "आपकी रिपोर्ट में {l} {v} है"]).format(l=lab, v=v)
            labs = [f"{lab} {v}"]
        if text in seen:
            continue
        seen.add(text)
        rows.append({"id": f"g{len(rows) + 1:03d}", "text": text, "drugs": ";".join(drugs),
                     "doses": ";".join(doses), "symptoms": ";".join(symptoms), "labs": ";".join(labs)})
    return rows


def validate(rows):
    return [(r["id"], c, e) for r in rows for c in ("drugs", "doses", "symptoms", "labs")
            for e in filter(None, r[c].split(";")) if e not in r["text"]]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--template", default="data/medical/sentences_template.csv")
    p.add_argument("--out", default="data/medical/medical_set.csv")
    p.add_argument("--n-generated", type=int, default=80)
    a = p.parse_args()
    with open(a.template, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows += generate(a.n_generated)
    bad = validate(rows)
    assert not bad, f"entities not found in text: {bad}"
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "drugs", "doses", "symptoms", "labs"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} sentences -> {a.out}")
