"""Percentile bootstrap (2000 resamples of test images, thresholds fixed) for the headline open-set metrics."""
import json

import numpy as np

from src.data import load_metadata
from src.open_set import auroc_unknown_vs_known, max_confidence

rng = np.random.RandomState(42)
df = load_metadata(); y = df[df.split == "test"].label.values
res = json.load(open("artifacts/metrics/open_set_results.json")); out = {}
for k in ("hog_svm", "mobilenet"):
    P = np.load(f"artifacts/predictions/{k}.npz"); thr = res[k]["threshold"]
    kp, up = P["test_probs"], P["unknown_probs"]; kc, uc = max_confidence(kp), max_confidence(up)
    vals = {m: [] for m in ("known_accuracy", "known_coverage", "unknown_rejection", "auroc")}
    for _ in range(2000):
        i, j = rng.randint(0, len(kc), len(kc)), rng.randint(0, len(uc), len(uc))
        vals["known_accuracy"].append((kp[i].argmax(1) == y[i]).mean()); vals["known_coverage"].append((kc[i] >= thr).mean())
        vals["unknown_rejection"].append((uc[j] < thr).mean()); vals["auroc"].append(auroc_unknown_vs_known(kc[i], uc[j]))
    out[k] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in vals.items()}
json.dump(out, open("artifacts/metrics/bootstrap_ci.json", "w"), indent=2)
for k, d in out.items():
    print(k, {m: [round(x, 3) for x in v] for m, v in d.items()})
