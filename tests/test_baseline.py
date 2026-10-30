import json
from src.baseline import read_manifest, FORCED


def test_read_manifest(tmp_path):
    (tmp_path / "test.jsonl").write_text(
        json.dumps({"audio": "a.wav", "text": "नमस्ते hello", "lang": "hinglish"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    rows = read_manifest(tmp_path, "test")
    assert rows[0]["lang"] == "hinglish" and FORCED[rows[0]["lang"]] == "hi"


def test_utt_stats_flags_hallucination():
    from src.metrics import utt_stats
    s = utt_stats(["hello world", "good morning"], ["hello world", "good morning " * 10])
    assert s["halluc_rate"] == 0.5
    assert 0.0 <= s["median_utt_wer"] <= 5.0
