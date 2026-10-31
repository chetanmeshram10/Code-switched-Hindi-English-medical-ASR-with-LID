# Code-switched Hindi-English Medical ASR with Language ID

Whisper-small fine-tuned with LoRA for Hindi-English (Hinglish) code-switched speech, plus an utterance-level
language-ID head and a clinically-oriented evaluation (drug / dose / symptom / lab-term accuracy).

> Research prototype. Not a medical device. Trained and evaluated on public data and synthetic speech only; no patient data.

## Why this project
Doctors in India mix Hindi and English mid-sentence ("Metformin 500 mg सुबह खाने के बाद लीजिए"). Off-the-shelf
Whisper handles this poorly: zero-shot Whisper-small hallucinates repetition loops on more than half of short
Hinglish clips (see results), and it writes English terms in the wrong script.

## What is in this repo
| Component | File | What it does |
|---|---|---|
| Normalizer + metrics | `src/normalize.py`, `src/metrics.py` | Script-aware text normalization (nukta, digits, danda), WER/CER/MER, hallucination rate |
| Data pipeline | `src/data.py` | Speaker-disjoint train/val/test manifests from public data (see Data) |
| Zero-shot baselines | `src/baseline.py` | Whisper small/medium, auto vs forced language |
| LoRA fine-tuning | `src/train.py` | Whisper-small + LoRA (rank 32), optional speed/noise/SpecAugment |
| Language ID | `src/lid.py` | hi / en / hinglish head on a frozen Whisper encoder, vs Whisper's built-in LID |
| Medical probe set | `src/make_medical_set.py`, `src/tts_medical.py`, `src/medical_eval.py` | 120 Hinglish clinical sentences, synthetic speech, entity-level metrics |
| Report + demo | `src/make_report.py`, `app.py` | Auto-generated result tables; Gradio demo with language tags and medical-term highlighting |

## Results
<!-- RESULTS:START -->
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
<!-- RESULTS:END -->

## Data
- **Training / evaluation speech:** [`addyo07/noisy-hinglish-asr`](https://huggingface.co/datasets/addyo07/noisy-hinglish-asr)
  (clean subset only). It aggregates Hindi (IndicVoices, FLEURS), English (AMI, FLEURS) and code-switched Hindi-English
  (MUCS). Splits are speaker-disjoint (train speakers are excluded from validation/test). Please check the upstream
  licences before redistributing models trained on it.
- **Label noise:** some Hinglish transcripts glue Devanagari and Latin letters ("अरैarray"); these clips are dropped.
- **Medical probe set:** 40 hand-written + 80 template-generated Hinglish clinical sentences with annotated entities
  (`data/medical/medical_set.csv`). Audio is generated with neural TTS (two Hindi voices), so it measures
  *terminology handling*, not real-world robustness.

## Method
- Transcript convention: Hindi in Devanagari, English words (incl. drug names) in Latin script, ASCII digits.
- Fine-tuning: LoRA (r=32, all attention + feed-forward projections, 13M trainable of 255M parameters), 3 epochs,
  language-token conditioning (Hinglish is labelled `hi`), fp16 on a single T4.
- Decoding: greedy, with a token budget proportional to clip length (removes most runaway repetition loops).
- Metrics: WER/CER/MER on normalized text; hallucination rate = share of clips whose output has >2x the reference
  word count; medical entity recall (strict, word-boundary match: a Devanagari spelling of a Latin drug name counts
  as a miss); "critical-exact" = every drug and dose in a sentence correct.

## Reproduce (Kaggle, GPU T4)
```bash
git clone https://github.com/chetanmeshram10/Code-switched-Hindi-English-medical-ASR-with-LID && cd Code-switched-Hindi-English-medical-ASR-with-LID
pip install -q -r requirements-kaggle.txt
python -m src.data --out /kaggle/working/data --n-hi 3000 --n-en 1000 --n-hinglish 4000 --n-val 100 --n-test 100
python -m src.baseline --data /kaggle/working/data --models openai/whisper-small openai/whisper-medium
python -m src.train --name e2 --langs hi en hinglish --epochs 3
python -m src.make_medical_set && python -m src.tts_medical --engine edge --out /kaggle/working/medical
python -m src.medical_eval --systems small=openai/whisper-small medium=openai/whisper-medium \
    e2=openai/whisper-small:/kaggle/working/checkpoints/e2/adapter
python -m src.lid --data /kaggle/working/data --medical-manifest /kaggle/working/medical/manifest.jsonl
python -m src.make_report
```
Demo: `ADAPTER=<path-or-hf-repo> python app.py`

## Limitations (please read)
- Test sets are small (100 clips per language) and the held-out pool has only ~10 speakers for English and Hinglish,
  so confidence intervals are wide; differences of a few WER points are not significant.
- The medical probe set is synthetic speech from two TTS voices: clean audio, no accents, no noise. It does not
  replace a set of real clinician recordings (planned future work).
- Hindi/English training data is conversational and tutorial speech, not clinical dialogue.
- The LID head is utterance-level; word-level code-switch tagging in the demo is script-based (Devanagari vs Latin),
  not a learned model.
- Language classes are confounded with source corpus (MUCS tutorials = Hinglish, IndicVoices = Hindi, AMI/FLEURS = English), so language-ID accuracy on held-out clips from the same corpora is probably optimistic; the head may partly recognise recording conditions. As a sanity check it is also run on the synthetic medical clips (a different domain).
- Single run per configuration (no seeds/variance); no clinical validation.

## Future work
Real recorded clinical test set, word-level LID, speaker diarization + LLM summarization wrapper, noise-robustness
study using the dataset's noisy splits, larger LoRA ablations (e1: no code-switched data; e3: augmentation).

## Privacy
Only public datasets and synthetic speech. No patient data.
