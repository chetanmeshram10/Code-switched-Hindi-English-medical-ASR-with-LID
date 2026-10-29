"""Transcript normalization for Hindi-English (Hinglish) ASR evaluation.

Transcript convention (document this in the README):
  - Hindi words  -> Devanagari
  - English words (incl. drug names) -> Latin script, lowercased for scoring
  - Digits -> ASCII digits
The SAME normalizer must be applied to references and hypotheses.
"""
import re
import unicodedata

NUKTA = "\u093C"
CHANDRABINDU = "\u0901"
ANUSVARA = "\u0902"
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_SPACE_RE = re.compile(r"\s+")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    """Normalize a transcript for WER/CER scoring."""
    text = unicodedata.normalize("NFD", text)        # split precomposed nukta letters (ज़ -> ज + ़)
    text = text.replace(NUKTA, "")                    # merge nukta variants (ज़ == ज)
    text = text.replace(CHANDRABINDU, ANUSVARA)       # ँ -> ं
    text = unicodedata.normalize("NFC", text)
    text = text.translate(DEVANAGARI_DIGITS)          # ५०० -> 500
    text = text.lower()                               # Latin only; no-op for Devanagari
    text = text.replace("।", " ").replace("॥", " ")   # danda
    # punctuation/symbols -> space (category-based: \w would wrongly drop Devanagari vowel signs)
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return _SPACE_RE.sub(" ", text).strip()


def word_script(word: str) -> str:
    """Label a word 'hi' (Devanagari), 'en' (Latin) or 'other' (digits etc.).
    Reused in Phase 5 to auto-generate language-ID labels."""
    if _DEVANAGARI_RE.search(word):
        return "hi"
    if _LATIN_RE.search(word):
        return "en"
    return "other"


def script_labels(text: str):
    """Return [(word, lang)] for a normalized transcript."""
    return [(w, word_script(w)) for w in normalize(text).split()]
