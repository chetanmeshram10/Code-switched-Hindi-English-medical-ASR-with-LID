"""Clinically-oriented metrics on top of WER: entity recall (drugs/symptoms/labs), dose accuracy,
and 'critical exact' (every drug AND dose in a sentence transcribed correctly).
Matching is on normalized text with word boundaries, so a Devanagari spelling of a Latin drug name
counts as a miss (strict by design; discussed in the README)."""
import re

from .normalize import normalize

_DIGIT_LETTER = re.compile(r"(?<=\d)(?=[a-z])")
KINDS = ("drugs", "doses", "symptoms", "labs")


def prep(text: str) -> str:
    return " " + _DIGIT_LETTER.sub(" ", normalize(text)) + " "


def _split(cell):
    return [e for e in (cell or "").split(";") if e] if isinstance(cell, str) else list(cell or [])


def summarize(rows, hyps):
    tot = {k: 0 for k in KINDS}
    hit = {k: 0 for k in KINDS}
    crit_n = crit_ok = 0
    for r, h in zip(rows, hyps):
        hp = prep(h)
        found = {}
        for k in KINDS:
            for e in _split(r.get(k)):
                ok = prep(e) in hp
                tot[k] += 1
                hit[k] += ok
                found.setdefault(k, []).append(ok)
        if r.get("drugs"):
            crit_n += 1
            crit_ok += all(found.get("drugs", [])) and all(found.get("doses", [True]))
    safe = lambda a, b: round(a / b, 4) if b else None
    all_t, all_h = sum(tot.values()), sum(hit.values())
    return {"drug_recall": safe(hit["drugs"], tot["drugs"]), "dose_acc": safe(hit["doses"], tot["doses"]),
            "symptom_recall": safe(hit["symptoms"], tot["symptoms"]), "lab_recall": safe(hit["labs"], tot["labs"]),
            "entity_recall": safe(all_h, all_t), "critical_exact": safe(crit_ok, crit_n)}
