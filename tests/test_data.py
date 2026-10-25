import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import build_metadata, check_image, discover_images, stratified_split


def _make(tmp_path, folder, n, seed=0, dup=False):
    d = tmp_path / folder
    d.mkdir(parents=True)
    rng = np.random.RandomState(seed)
    for i in range(n):
        Image.fromarray(rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)).save(d / f"{i}.jpg")
    if dup:
        (d / "dup.jpg").write_bytes((d / "0.jpg").read_bytes())
    return d


def _cfg(tmp_path):
    return {"seed": 42, "raw_dir": str(tmp_path), "known": {"A": "A_h", "B": "B_h"}, "unknown": {"U": "U_h"},
            "max_per_known": 20, "max_per_unknown": 10, "split": {"train": .7, "val": .15, "test": .15}}


def test_discovery_and_missing_folder(tmp_path):
    _make(tmp_path, "A_h", 3)
    assert len(discover_images(tmp_path / "A_h")) == 3
    with pytest.raises(FileNotFoundError):
        discover_images(tmp_path / "nope")


def test_corrupt_image_detected_and_excluded(tmp_path):
    d = _make(tmp_path, "A_h", 25)
    (d / "bad.jpg").write_bytes(b"not an image")
    assert check_image(d / "bad.jpg")[0] is False
    _make(tmp_path, "B_h", 25, seed=1); _make(tmp_path, "U_h", 12, seed=2)
    df, log = build_metadata(_cfg(tmp_path))
    assert len(log["corrupt"]) == 1 and not df.path.str.contains("bad.jpg").any()


def test_exact_duplicates_removed(tmp_path):
    _make(tmp_path, "A_h", 25, dup=True); _make(tmp_path, "B_h", 25, seed=1); _make(tmp_path, "U_h", 12, seed=2)
    df, log = build_metadata(_cfg(tmp_path))
    assert len(log["exact_duplicates_removed"]) == 1 and df.sha256.is_unique


def test_split_reproducible_stratified_and_unknown_isolated(tmp_path):
    _make(tmp_path, "A_h", 30); _make(tmp_path, "B_h", 30, seed=1); _make(tmp_path, "U_h", 12, seed=2)
    cfg = _cfg(tmp_path)
    d1, _ = build_metadata(cfg); d2, _ = build_metadata(cfg)
    pd.testing.assert_frame_equal(d1, d2)
    assert set(d1[d1.role == "unknown"].split) == {"unknown_test"}
    assert not (d1[d1.role == "known"].split == "unknown_test").any()
    assert (d1[d1.role == "unknown"].label == -1).all()
    for sp in ("A", "B"):
        assert set(d1[d1.species == sp].split) == {"train", "val", "test"}
    assert d1.path.is_unique
