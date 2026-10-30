import io
import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
from src.data import to_16k, SR, _split_of, iter_rows, speaker_set, split_heldout, _GLUED


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


def test_split_of():
    assert _split_of(("data", "train", "0000.parquet")) == "train"
    assert _split_of(("data", "val", "0000.parquet")) == "validation"
    assert _split_of(("data", "benchmark", "hinglish", "0.parquet")) is None


def _write(tmp_path):
    rows = {"transcription": list("abcdef"),
            "language": ["hinglish"] * 6,
            "condition": ["clean", "clean", "clean", "clean", "synthetic_noise", "clean"],
            "source": ["mucs"] * 6,
            "speaker_id": ["s1", "s2", "s3", "s4", "s1", "s1"]}
    f = tmp_path / "x.parquet"
    pq.write_table(pa.table(rows), f)
    return f


def test_filters_and_speakers(tmp_path):
    f = _write(tmp_path)
    got = list(iter_rows([f], "hinglish", "clean", 10))
    assert sorted(r["transcription"] for r in got) == ["a", "b", "c", "d", "f"]  # noisy row 'e' excluded
    got = list(iter_rows([f], "hinglish", "clean", 10, speakers={"s1"}))
    assert sorted(r["transcription"] for r in got) == ["a", "f"]


def test_heldout_speakers_disjoint(tmp_path):
    f = _write(tmp_path)
    assert speaker_set([f], "hinglish", "clean") == {"s1", "s2", "s3", "s4"}
    p = split_heldout([f], "hinglish", "clean", exclude={"s4"})
    assert p["validation"][1].isdisjoint(p["test"][1])
    assert "s4" not in p["validation"][1] | p["test"][1]


def test_glued_script_detector():
    assert _GLUED.search("एक अरैarray के")
    assert not _GLUED.search("इस tutorial में module का")


def test_multi_rowgroup_streaming_with_audio(tmp_path):
    n = 7
    audio = [{"bytes": _wav_bytes(16000, 0.1), "path": f"{i}.wav"} for i in range(n)]
    tbl = pa.table({"transcription": [f"t{i}" for i in range(n)], "audio": audio,
                    "language": ["hi", "en"] * 3 + ["hi"],
                    "condition": ["clean"] * n, "source": ["x"] * n,
                    "speaker_id": [f"s{i}" for i in range(n)]})
    f = tmp_path / "big.parquet"
    pq.write_table(tbl, f, row_group_size=2)
    got = list(iter_rows([f], "hi", "clean", 100))
    assert sorted(r["transcription"] for r in got) == ["t0", "t2", "t4", "t6"]
    assert all(isinstance(r["audio"]["bytes"], (bytes, bytearray)) for r in got)
