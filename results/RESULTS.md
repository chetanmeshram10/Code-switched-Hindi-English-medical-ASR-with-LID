### ASR on held-out public test sets (test split, forced language)

| System | Hindi WER | English WER | Hinglish WER | Hinglish CER | Hinglish hallucination rate |
|---|---|---|---|---|---|
| whisper-small (zero-shot) | 1.19 | 0.23 | 2.14 | 1.46 | 52% |
| whisper-medium (zero-shot) | 0.65 | 0.21 | 0.81 | 0.69 | 13% |
| **e2** (LoRA whisper-small) | 0.30 | 0.25 | 0.38 | 0.31 | 2% |

### Does language detection matter? (WER, auto vs forced language token)

| System | Hindi (auto / forced) | English (auto / forced) | Hinglish (auto / forced) |
|---|---|---|---|
| whisper-small (zero-shot) | 1.19 / 1.19 | 0.40 / 0.23 | 2.02 / 2.14 |
| whisper-medium (zero-shot) | 0.68 / 0.65 | 0.49 / 0.21 | 0.78 / 0.81 |
| **e2** (LoRA whisper-small) | 0.30 / 0.30 | 0.25 / 0.25 | 0.38 / 0.38 |

### Medical probe set (synthetic speech, forced Hindi decoding)

| System | WER | Drug recall | Dose accuracy | Symptom recall | Lab recall | Critical-exact |
|---|---|---|---|---|---|---|
| e2 | 0.25 | 24% | 52% | 75% | 31% | 20% |
| small | 0.63 | 0% | 0% | 36% | 9% | 0% |
| medium | 0.43 | 11% | 33% | 52% | 24% | 11% |

### Utterance-level language ID

| Metric | Value |
|---|---|
| Whisper built-in LID, hi/en collapsed accuracy | 92% |
| Our head, hi/en collapsed accuracy | 100% |
| Our head, 3-class accuracy (hi / en / hinglish) | 99% |
| Our head, macro-F1 | 0.99 |
| F1 (hi) | 0.99 |
| F1 (en) | 1.00 |
| F1 (hinglish) | 0.99 |
| Head on synthetic medical clips (all Hinglish): predicted hi / en / hinglish | 33% / 0% / 67% |

Confusion matrix (test set):

| gold \ predicted | hi | en | hinglish |
|---|---|---|---|
| hi | 98 | 0 | 2 |
| en | 0 | 100 | 0 |
| hinglish | 0 | 0 | 100 |
