import numpy as np
from src.train import augment_wave


def test_augment_changes_signal_and_dtype():
    rng = np.random.default_rng(0)
    wav = (0.1 * np.sin(np.linspace(0, 200, 16000))).astype(np.float32)
    outs = [augment_wave(wav, rng) for _ in range(20)]
    assert all(o.dtype == np.float32 and o.ndim == 1 and len(o) > 0 for o in outs)
    assert any(not np.array_equal(o, wav) for o in outs)
