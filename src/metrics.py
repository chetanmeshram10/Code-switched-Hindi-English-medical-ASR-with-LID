"""WER / CER / MER with the shared normalizer."""
import jiwer
from .normalize import normalize


def score(refs, hyps):
    refs_n = [normalize(r) for r in refs]
    hyps_n = [normalize(h) for h in hyps]
    return {
        "wer": jiwer.wer(refs_n, hyps_n),
        "cer": jiwer.cer(refs_n, hyps_n),
        "mer": jiwer.mer(refs_n, hyps_n),
    }
