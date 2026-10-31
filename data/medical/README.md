> **Status:** the results in the root README use a *synthetic* probe set (`medical_set.csv`, spoken by TTS voices; see `src/tts_medical.py`). The recording protocol below is for a planned real-speech extension and has not been run.

# Medical Hinglish test set (self-recorded)

Purpose: a small, held-out, REAL-speech test set of doctor-style Hinglish sentences with drug names,
doses, symptoms and lab terms. It is the differentiator of this project. **Never train on it.**

## Transcript convention (same as the rest of the repo)
- Hindi words in Devanagari, English/drug/lab terms in Latin script.
- Doses and numbers as digits with a unit ("650 mg", "2 puff").
- No punctuation needed (the scorer strips it).

## CSV columns
`id, text, drugs, doses, symptoms, labs` - entity columns are `;`-separated and must appear in `text`
exactly as written (used for Medical Entity Recall and dosage accuracy in Phase 6).

## Target size and recording protocol
1. Write ~200 sentences in total (this file has 40 to copy the pattern from; vary drugs, doses, symptoms).
   Keep 8-20 words each; roughly 40% with a drug+dose, 30% symptoms, 30% labs/instructions.
2. Record with 5+ volunteers (mixed gender/accent), ~40 sentences each. Get consent from every speaker
   and tell them the audio will be published as an evaluation set.
3. Quiet room, phone voice recorder is fine. Export mono WAV (16 kHz ideal; any rate is resampled).
4. File names: `spk01_m001.wav` (speaker id + sentence id). Put them in `data/medical/audio/`
   (audio is git-ignored; publish it on Hugging Face instead).
5. Do not use real patient data. Do not use these sentences for TTS training data.

## Training-side synthetic data (optional, later)
Write a DIFFERENT set of sentences for TTS-generated training data so train and test never overlap.
