import numpy as np

from src.data_prep import clean, split_indices


def test_splits_do_not_overlap_and_cover_everything():
    idx = split_indices(1000, n_val=100, n_test=100, seed=1234)
    val, test, train = (set(idx[k].tolist()) for k in ("val", "test", "train"))
    assert len(val) == 100 and len(test) == 100 and len(train) == 800
    assert not (val & test) and not (val & train) and not (test & train)
    assert val | test | train == set(range(1000))


def test_split_is_deterministic():
    a = split_indices(1000, 100, 100, seed=1234)
    b = split_indices(1000, 100, 100, seed=1234)
    c = split_indices(1000, 100, 100, seed=1)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["val"], c["val"])


def test_clean_rules():
    records = [
        {"story": "  A cat sat.\n", "source": "GPT-4"},
        {"story": "A cat sat.", "source": "GPT-3.5"},     # duplicate after strip
        {"story": "   ", "source": "GPT-4"},               # empty
        {"story": "Itâ€™s fun.", "source": "GPT-4"},  # mojibake
        {"story": "She said “hi”.", "source": "GPT-3.5"},  # curly quotes are fine
    ]
    kept, removed = clean(records)
    assert [s["story"] for s in kept] == ["A cat sat.", "She said “hi”."]
    assert kept[0]["source"] == "GPT-4"
    assert removed == {"empty": 1, "mojibake": 1, "duplicate": 1}
