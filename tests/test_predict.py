from pathlib import Path

import pytest

from src.open_set import UNKNOWN_LABEL

needs_artifacts = pytest.mark.skipif(not (Path("artifacts/checkpoints/mobilenet_v3_small.pth").exists()
                                          and Path("artifacts/metrics/open_set_results.json").exists()),
                                     reason="run training + evaluation first")


@needs_artifacts
def test_inference_output_format_and_consistency_with_saved_scores():
    import pandas as pd
    from src.predict import LeafPredictor
    pr = LeafPredictor("mobilenet")
    d = pd.read_csv("artifacts/metrics/mobilenet_predictions.csv").iloc[[0, -1]]
    out = pr.predict_paths(d.path.tolist())
    for r, (_, row) in zip(out, d.iterrows()):
        assert set(r) == {"predicted", "confidence", "outcome", "accepted", "threshold"}
        assert r["predicted"] in pr.classes and 0 < r["confidence"] <= 1
        assert r["outcome"] == (r["predicted"] if r["accepted"] else UNKNOWN_LABEL)
        assert r["predicted"] == row.pred and abs(r["confidence"] - row.conf) < 1e-3  # same pipeline as evaluation
