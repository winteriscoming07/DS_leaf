"""SYNTHETIC perturbation test: brightness / blur applied in memory to copies of known TEST images.
Thresholds stay frozen from validation. Not evidence about real field photography."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageEnhance, ImageFilter

from src.data import load_metadata
from src.predict import LeafPredictor
from src.preprocessing import load_rgb

PERTURB = {"brightness": [0.4, 0.7, 1.0, 1.3, 1.6], "blur_radius": [0, 1, 2, 4]}


def apply(img, kind, v):
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(v)
    return img.filter(ImageFilter.GaussianBlur(v)) if v > 0 else img


def main():
    df = load_metadata(); te = df[df.split == "test"]
    imgs, y = [load_rgb(p) for p in te.path], te.label.values
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, color in (("hog_svm", "#c0762b"), ("mobilenet", "#2b6cc0")):
        pr = LeafPredictor(key); out[key] = {}
        for ax, (kind, levels) in zip(axes, PERTURB.items()):
            rows = []
            for v in levels:
                P = pr.probs([apply(im, kind, v) for im in imgs])
                conf, pred = P.max(1), P.argmax(1); acc = conf >= pr.threshold
                rows.append(dict(level=v, accuracy=float((pred == y).mean()), coverage=float(acc.mean()),
                                 accuracy_on_accepted=float((pred[acc] == y[acc]).mean()) if acc.any() else None))
            out[key][kind] = rows
            ax.plot(levels, [r["accuracy"] for r in rows], "-o", color=color, label=f"{key} accuracy")
            ax.plot(levels, [r["coverage"] for r in rows], "--s", color=color, label=f"{key} coverage")
            ax.set_xlabel(kind); ax.set_ylim(-.02, 1.05); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7); axes[0].set_title("Brightness (1.0 = original)"); axes[1].set_title("Gaussian blur radius (px)")
    fig.suptitle("Synthetic perturbation of known test images (thresholds frozen)", fontsize=10)
    plt.tight_layout(); plt.savefig("artifacts/figures/stress_test.png", dpi=140); plt.close()
    json.dump(out, open("artifacts/metrics/stress_test.json", "w"), indent=2)
    for k, d in out.items():
        for kind, rows in d.items():
            print(k, kind, [(r["level"], round(r["accuracy"], 3), round(r["coverage"], 3)) for r in rows])


if __name__ == "__main__":
    main()
