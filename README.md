# MedASR: Code-switched Hindi-English Medical ASR with Language ID

Fine-tune Whisper (LoRA) for Hindi-English code-switched clinical speech, add word/segment-level
language ID, and evaluate on a custom medical-term test set.

## Transcript convention
Hindi in Devanagari, English (incl. drug names) in Latin script, ASCII digits.
See `src/normalize.py`. The same normalizer is applied to references and hypotheses.

## Status
- [x] Phase 1: normalizer + metrics
- [x] Phase 2: data loader (`src/data.py`)
- [ ] Phase 3: baselines
- [ ] Phase 4: LoRA fine-tuning
- [ ] Phase 5: LID head
- [ ] Phase 6: medical metrics + error analysis
- [ ] Phase 7: Gradio demo

## Results
_Filled in as experiments finish._

## Privacy
Only public or synthetic data and consented volunteer recordings. No patient data.

## Limitations
_To be completed._
