"""Model A: HOG + RBF-SVM (probability estimates via Platt scaling). Fitted on known-train only."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.svm import SVC

from src.data import load_config, load_metadata
from src.preprocessing import hog_features, load_rgb


def features(paths, size):
    return np.stack([hog_features(load_rgb(p), size) for p in paths])


def main():
    cfg = load_config()
    df = load_metadata()
    classes = list(cfg["known"])
    sz = cfg["image_size_hog"]
    tr = df[df.split == "train"]
    assert (tr.role == "known").all(), "unknown species must never be in training"
    t0 = time.time()
    Xtr = features(tr.path, sz)
    svm = SVC(C=cfg["svm"]["C"], kernel=cfg["svm"]["kernel"], gamma=cfg["svm"]["gamma"],
              probability=True, random_state=cfg["seed"])
    svm.fit(Xtr, tr.label.values)
    fit_s = time.time() - t0
    out, timing = {}, {}
    for name, split in (("val", "val"), ("test", "test"), ("unknown", "unknown_test")):
        sub = df[df.split == split]
        t = time.time()
        out[f"{name}_probs"] = svm.predict_proba(features(sub.path, sz))
        timing[name] = (time.time() - t) / len(sub)
    Path("artifacts/predictions").mkdir(parents=True, exist_ok=True)
    Path("artifacts/checkpoints").mkdir(parents=True, exist_ok=True)
    np.savez("artifacts/predictions/hog_svm.npz", **out)
    joblib.dump(svm, "artifacts/checkpoints/hog_svm.joblib")
    info = {"model": "HOG+SVM", "n_train": int(len(tr)), "feature_dim": int(Xtr.shape[1]), "fit_seconds": round(fit_s, 1),
            "inference_seconds_per_image_incl_features": {k: round(v, 5) for k, v in timing.items()},
            "svm": cfg["svm"], "hog_image_size": sz, "train_accuracy": float(svm.score(Xtr, tr.label.values)),
            "sklearn": sklearn.__version__, "seed": cfg["seed"]}
    Path("artifacts/logs").mkdir(exist_ok=True)
    json.dump(info, open("artifacts/logs/hog_svm_train.json", "w"), indent=2)
    print(json.dumps(info, indent=1))
    va = df[df.split == "val"]
    print("val acc (for information):", float((out["val_probs"].argmax(1) == va.label.values).mean()))


if __name__ == "__main__":
    main()
