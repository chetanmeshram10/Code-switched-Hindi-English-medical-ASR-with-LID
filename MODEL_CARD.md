---
language: [hi, en]
license: mit
base_model: openai/whisper-small
library_name: peft
tags: [automatic-speech-recognition, whisper, lora, code-switching, hinglish, medical]
---
# whisper-small-hinglish-lora

LoRA adapter for `openai/whisper-small` for Hindi-English code-switched speech (Devanagari for Hindi words, Latin
script for English words).

## Intended use
Research on code-switched ASR. **Not** for clinical decision-making.

## Training data
Clean subset of `addyo07/noisy-hinglish-asr` (Hindi, English and code-switched Hindi-English speech; upstream
sources include MUCS, IndicVoices, AMI and FLEURS). Check upstream licences before reuse. ~15 hours of speech.

## Training
LoRA r=32 / alpha=64 on q,k,v,out_proj,fc1,fc2; lr 5e-4, effective batch 16, 3 epochs, fp16.

## Evaluation
See the README results tables (public held-out test sets, synthetic medical probe set, language-ID head).

## Limitations
Small test sets, few held-out speakers, synthetic medical probe audio, no noise or accent robustness study, no
clinical validation. May hallucinate repeated text on very short or silent audio.

## Usage
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "<your-username>/whisper-small-hinglish-lora").merge_and_unload()
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
# generate(..., language="hi", task="transcribe")
```
