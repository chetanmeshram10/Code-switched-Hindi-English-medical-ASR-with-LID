from src.normalize import normalize, script_labels


def test_nukta_and_chandrabindu():
    assert normalize("ज़रूर") == normalize("जरूर")
    assert normalize("आँख") == normalize("आंख")


def test_punct_and_case():
    assert normalize("Take Metformin, 500 mg.") == "take metformin 500 mg"


def test_devanagari_digits_and_danda():
    assert normalize("५०० mg दीजिए।") == "500 mg दीजिए"


def test_code_switch_labels():
    labels = script_labels("मरीज़ को Metformin 500 mg दीजिए")
    assert ("metformin", "en") in labels
    assert ("दीजिए", "hi") in labels
    assert ("500", "other") in labels
