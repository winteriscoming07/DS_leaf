import pandas as pd
import pytest

from src.grouped_split import blockwise_split, min_cross_split_gap, parse_name

RATIOS = {"train": .7, "val": .15, "test": .15}


def test_parse_name_variants():
    assert parse_name("x/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG") == ("frame:RS", 5672)
    assert parse_name("00031d74-076e-4aef-b040-e068cd3576eb___R.S_HL 8315 copy 2.jpg") == ("frame:R.S", 8315)
    assert parse_name("a/000146ff-92a4-4db6-90ad-8fce2ae4fddd___GH_HL Leaf 259.1.JPG") == ("leaf", 259)
    with pytest.raises(ValueError):
        parse_name("cat.jpg")


def _toy(n_per=120):
    rows = []
    for sp, prefix in (("A", "RS"), ("B", "Mt")):
        for i in range(n_per):
            rows.append(dict(path=f"d/{i:036d}___{prefix}_HL {1000 + 3 * i}.JPG", species=sp, role="known", split="train", label=0))
    rows += [dict(path=f"u/{i:036d}___JR_HL {i}.JPG", species="U", role="unknown", split="unknown_test", label=-1) for i in range(20)]
    return pd.DataFrame(rows)


def test_deterministic_unknown_untouched_and_no_group_straddles_splits():
    df = _toy()
    a, b = blockwise_split(df, RATIOS, 42), blockwise_split(df, RATIOS, 42)
    pd.testing.assert_frame_equal(a, b)
    assert (a[a.role == "unknown"].split == "unknown_test").all()
    k = a[(a.role == "known") & (a.split.isin(["train", "val", "test"]))]
    assert (k.groupby("group").split.nunique() == 1).all()
    assert set(k.split) == {"train", "val", "test"}


def test_guard_band_enforces_minimum_frame_gap():
    a = blockwise_split(_toy(), RATIOS, 42, block=50, guard=5)
    assert (a.split == "guard_excluded").any()
    for pair in (("train", "val"), ("train", "test"), ("val", "test")):
        assert min_cross_split_gap(a, *pair) > 5


def test_leaf_ids_never_split_across_sets():
    rows = [dict(path=f"d/{i:036d}___GH_HL Leaf {i // 2}.{i % 2 + 1}.JPG", species="T", role="known", split="train", label=0) for i in range(100)]
    a = blockwise_split(pd.DataFrame(rows), RATIOS, 1)
    assert (a.groupby("group_num").split.nunique() == 1).all()
