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


def test_split_of_and_hinglish_filter(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.data import _split_of, iter_hinglish
    assert _split_of(("data", "train", "0000.parquet")) == "train"
    assert _split_of(("data", "val", "0000.parquet")) == "validation"
    assert _split_of(("data", "benchmark", "hinglish", "0.parquet")) is None
    rows = {"transcription": ["a", "b", "c"], "language": ["hinglish", "en", "hinglish"],
            "condition": ["clean", "clean", "noisy"]}
    f = tmp_path / "x.parquet"
    pq.write_table(pa.table(rows), f)
    got = list(iter_hinglish([f], "hinglish", "clean", 10))
    assert [r["transcription"] for r in got] == ["a"]
