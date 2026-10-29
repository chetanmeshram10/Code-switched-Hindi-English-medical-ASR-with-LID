import io
import numpy as np
import soundfile as sf
from src.data import to_16k, SR


def _wav_bytes(sr, secs=1.0, channels=1):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    x = 0.1 * np.sin(2 * np.pi * 220 * t)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV")
    return buf.getvalue()


def test_resample_and_mono():
    y = to_16k({"bytes": _wav_bytes(44100, 1.0, 2)})
    assert y.ndim == 1 and abs(len(y) - SR) < 50
