import numpy as np
import torch
from PIL import Image

from src.preprocessing import eval_transform, hog_features


def test_eval_transform_shape_dtype_deterministic():
    img = Image.fromarray(np.random.RandomState(0).randint(0, 255, (256, 256, 3), dtype=np.uint8))
    t = eval_transform(224)
    a, b = t(img), t(img)
    assert a.shape == (3, 224, 224) and a.dtype == torch.float32 and torch.equal(a, b)


def test_hog_features_fixed_length():
    f1 = hog_features(Image.new("RGB", (256, 256), "green"))
    f2 = hog_features(Image.new("RGB", (100, 80), "red"))
    assert f1.ndim == 1 and f1.shape == f2.shape and f1.dtype.kind == "f"
