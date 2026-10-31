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
