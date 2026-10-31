"""Gradio demo: Hinglish speech -> transcript with per-word language tags and highlighted medical terms.

Local / HF Space:
  ADAPTER=<hf-repo-id-or-local-dir>  BASE_MODEL=openai/whisper-small  python app.py
Without ADAPTER it runs the zero-shot base model (useful to compare against the fine-tuned adapter).
"""
import html
import os

import gradio as gr
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.make_medical_set import MED_TERMS
from src.normalize import normalize, word_script

BASE = os.environ.get("BASE_MODEL", "openai/whisper-small")
ADAPTER = os.environ.get("ADAPTER", "")

proc = WhisperProcessor.from_pretrained(BASE)
model = WhisperForConditionalGeneration.from_pretrained(BASE)
model.generation_config.forced_decoder_ids = None
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload()
model.eval()

STYLE = {"hi": "background:#dbeafe;color:#1e3a8a", "en": "background:#ffedd5;color:#9a3412", "other": "background:#f3f4f6;color:#374151"}


def tag_html(text):
    spans = []
    for w in text.split():
        lang = word_script(normalize(w)) if normalize(w) else "other"
        style = STYLE[lang]
        if normalize(w) in MED_TERMS:
            style += ";border:2px solid #dc2626;font-weight:700"
        spans.append(f'<span style="{style};padding:2px 6px;margin:2px;border-radius:6px;display:inline-block">{html.escape(w)}</span>')
    legend = ('<div style="font-size:12px;margin-top:8px">'
              '<span style="background:#dbeafe;padding:2px 6px;border-radius:6px">Hindi (Devanagari)</span> '
              '<span style="background:#ffedd5;padding:2px 6px;border-radius:6px">English (Latin)</span> '
              '<span style="border:2px solid #dc2626;padding:0 6px;border-radius:6px">medical term</span></div>')
    return "<div style='line-height:2.2'>" + " ".join(spans) + "</div>" + legend


def transcribe(path):
    if not path:
        return "", ""
    wav, _ = librosa.load(path, sr=16000, mono=True)
    wav = wav[:16000 * 30]
    feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features
    max_tokens = min(440, int(len(wav) / 16000 * 20) + 30)
    with torch.no_grad():
        ids = model.generate(feats, language="hi", task="transcribe", max_new_tokens=max_tokens)
    text = proc.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return text, tag_html(text)


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["upload", "microphone"], type="filepath", label="Hinglish speech (max 30 s)"),
    outputs=[gr.Textbox(label="Transcript"), gr.HTML(label="Language tags + medical terms")],
    title="Code-switched Hindi-English medical ASR",
    description="LoRA-fine-tuned Whisper-small. Research demo only: not a medical device, do not use for clinical decisions.",
)

if __name__ == "__main__":
    demo.launch()
