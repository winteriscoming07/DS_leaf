import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.predict import open_upload

APP = str(Path(__file__).resolve().parent.parent / "app.py")
needs_artifacts = pytest.mark.skipif(not Path("artifacts/checkpoints/mobilenet_v3_small.pth").exists(), reason="run training first")


def _png(mode, size=(40, 30), **kw):
    b = io.BytesIO(); Image.new(mode, size, **kw).save(b, "PNG"); b.seek(0); return b


def test_open_upload_converts_modes_to_rgb():
    for mode in ("RGBA", "L", "P", "RGB"):
        im = open_upload(_png(mode))
        assert im.mode == "RGB" and im.size == (40, 30)


def test_open_upload_transparent_pixels_become_white():
    assert open_upload(_png("RGBA", color=(0, 0, 0, 0))).getpixel((1, 1)) == (255, 255, 255)


def test_open_upload_rejects_garbage_and_empty():
    for data in (b"not an image", b""):
        with pytest.raises(ValueError, match="could not be read"):
            open_upload(io.BytesIO(data))


@needs_artifacts
def test_app_upload_workflow_known_and_unknown_and_bad_file():
    from streamlit.testing.v1 import AppTest
    d = pd.read_csv("artifacts/metrics/mobilenet_predictions.csv")
    cases = [d[(d.role == "known") & d.accepted].iloc[0], d[(d.role == "unknown") & ~d.accepted].iloc[0]]
    for row in cases:
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception and not at.error
        assert len(at.file_uploader) == 1
        _upload(at, Path(row.path).read_bytes(), "leaf.jpg")
        at.run()
        assert not at.exception
        text = " ".join(e.value for e in at.success) + " ".join(e.value for e in at.warning)
        assert ("Unknown / Uncertain" in text) == (not row.accepted)
        assert any("not a botanical identification" in i.value for i in at.info)
    at = AppTest.from_file(APP, default_timeout=60).run()
    _upload(at, b"garbage", "bad.jpg"); at.run()
    assert not at.exception and any("could not be read" in e.value for e in at.error)


def _upload(at, data, name):
    at.file_uploader[0].upload(name, data, "image/jpeg")
