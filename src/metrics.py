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


def utt_stats(refs, hyps):
    """Robust companions to corpus WER (which a few runaway hallucinations can push above 1.0).
    median_utt_wer : median of per-utterance WER
    halluc_rate    : share of utterances whose hypothesis has > 2x the reference word count
    """
    import statistics
    wers, halluc, n = [], 0, 0
    for r, h in zip(refs, hyps):
        rn, hn = normalize(r), normalize(h)
        if not rn:
            continue
        n += 1
        wers.append(jiwer.wer(rn, hn) if hn else 1.0)
        if len(hn.split()) > 2 * len(rn.split()):
            halluc += 1
    return {"median_utt_wer": statistics.median(wers) if wers else 0.0,
            "halluc_rate": halluc / n if n else 0.0}
