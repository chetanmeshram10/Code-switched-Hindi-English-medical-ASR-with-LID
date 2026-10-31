"""Generate synthetic speech for the medical probe set (no recording needed).
Engines: edge-tts (Microsoft neural Hindi voices, 2 voices) or gTTS (Google, 1 voice).
Needs internet. Output: <out>/audio/<voice>_<id>.wav (16 kHz mono) + <out>/manifest.jsonl

  pip install edge-tts gTTS
  python -m src.tts_medical --engine edge --out /kaggle/working/medical
"""
import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import soundfile as sf

VOICES = {"edge": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"], "gtts": ["gtts-hi"]}


async def _edge(text, voice, path):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(str(path))


def synth(engine, text, voice, path):
    if engine == "edge":
        asyncio.run(_edge(text, voice, path))
    else:
        from gtts import gTTS
        gTTS(text, lang="hi").save(str(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/medical/medical_set.csv")
    p.add_argument("--out", default="/kaggle/working/medical")
    p.add_argument("--engine", choices=["edge", "gtts"], default="edge")
    p.add_argument("--limit", type=int, default=0, help="only first N sentences (debug)")
    a = p.parse_args()
    import librosa
    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    ok = fail = 0
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for r in rows:
            for voice in VOICES[a.engine]:
                mp3, wav = out / "audio" / f"{voice}_{r['id']}.mp3", out / "audio" / f"{voice}_{r['id']}.wav"
                for attempt in range(3):
                    try:
                        if not wav.exists():
                            synth(a.engine, r["text"], voice, mp3)
                            y, _ = librosa.load(str(mp3), sr=16000, mono=True)
                            sf.write(wav, y, 16000, subtype="PCM_16")
                            mp3.unlink(missing_ok=True)
                        dur = sf.info(wav).duration
                        mf.write(json.dumps({**r, "audio": f"audio/{wav.name}", "voice": voice, "lang": "hinglish",
                                             "duration": round(dur, 2)}, ensure_ascii=False) + "\n")
                        ok += 1
                        break
                    except Exception as e:  # network / decode hiccup
                        if attempt == 2:
                            fail += 1
                            print(f"[fail] {voice} {r['id']}: {type(e).__name__}: {str(e)[:100]}")
                        time.sleep(1.5)
    print(f"done: {ok} clips, {fail} failures -> {out}")


if __name__ == "__main__":
    main()
