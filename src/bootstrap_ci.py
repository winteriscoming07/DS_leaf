"""Percentile bootstrap (2000 resamples of test images, thresholds fixed) for the headline open-set metrics.
Also reports exact Clopper-Pearson intervals for the three proportions, because a percentile bootstrap of an
all-correct sample (e.g. 188/188) collapses to the degenerate interval [1, 1].

Limitations: images are resampled as if independent (specimen/capture-session clustering is ignored); the threshold,
the trained model and the train/val/test split are held fixed, so only test-sampling noise is captured."""
import json

import numpy as np
from scipy.stats import beta

from src.data import load_metadata
from src.open_set import auroc_unknown_vs_known, max_confidence
from src.paths import ART


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def main(n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    df = load_metadata(); y = df[df.split == "test"].label.values
    res = json.load(open(ART / "metrics/open_set_results.json")); out = {}
    for k in ("hog_svm", "mobilenet"):
        P = np.load(ART / f"predictions/{k}.npz"); thr = res[k]["threshold"]
        kp, up = P["test_probs"], P["unknown_probs"]; kc, uc = max_confidence(kp), max_confidence(up)
        vals = {m: [] for m in ("known_accuracy", "known_coverage", "unknown_rejection", "auroc")}
        for _ in range(n_boot):
            i, j = rng.randint(0, len(kc), len(kc)), rng.randint(0, len(uc), len(uc))
            vals["known_accuracy"].append((kp[i].argmax(1) == y[i]).mean()); vals["known_coverage"].append((kc[i] >= thr).mean())
            vals["unknown_rejection"].append((uc[j] < thr).mean()); vals["auroc"].append(auroc_unknown_vs_known(kc[i], uc[j]))
        out[k] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in vals.items()}
        out[k]["clopper_pearson_exact"] = {
            "known_accuracy": clopper_pearson(int((kp.argmax(1) == y).sum()), len(y)),
            "known_coverage": clopper_pearson(int((kc >= thr).sum()), len(kc)),
            "unknown_rejection": clopper_pearson(int((uc < thr).sum()), len(uc))}
    json.dump(out, open(ART / "metrics/bootstrap_ci.json", "w"), indent=2)
    for k, d in out.items():
        print(k, {m: [round(x, 3) for x in v] for m, v in d.items() if m != "clopper_pearson_exact"})
        print(k, "exact:", {m: [round(x, 3) for x in v] for m, v in d["clopper_pearson_exact"].items()})


if __name__ == "__main__":
    main()
