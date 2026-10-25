"""Open-set evaluation of both models from saved probabilities. Threshold uses VALIDATION known images only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.data import load_config, load_metadata
from src.open_set import (auroc_unknown_vs_known, combined_correctness, combined_risk_at_threshold, known_metrics,
                          max_confidence, risk_coverage, select_threshold, unknown_metrics)

MODELS = {"hog_svm": "HOG + SVM", "mobilenet": "MobileNetV3-Small"}
COLORS = {"hog_svm": "#c0762b", "mobilenet": "#2b6cc0"}
FIG, MET = Path("artifacts/figures"), Path("artifacts/metrics")


def evaluate_model(key, df, cfg):
    classes = list(cfg["known"])
    P = np.load(f"artifacts/predictions/{key}.npz")
    val, te, unk = df[df.split == "val"], df[df.split == "test"], df[df.split == "unknown_test"]
    thr = select_threshold(max_confidence(P["val_probs"]), cfg["threshold_percentile"])  # frozen here, before any test metric
    km = known_metrics(P["test_probs"], te.label.values, thr)
    um = unknown_metrics(P["unknown_probs"], thr)
    kc, uc = max_confidence(P["test_probs"]), max_confidence(P["unknown_probs"])
    res = {"threshold": thr, "known_test": km, "unknown": um, "auroc_unknown_positive": auroc_unknown_vs_known(kc, uc),
           "auroc_per_unknown_species": {s: auroc_unknown_vs_known(kc, uc[(unk.species == s).values]) for s in unk.species.unique()},
           "unknown_rejection_per_species": {s: unknown_metrics(P["unknown_probs"][(unk.species == s).values], thr)["rejection_rate"] for s in unk.species.unique()},
           "combined": combined_risk_at_threshold(P["test_probs"], te.label.values, P["unknown_probs"], thr),
           "val_acceptance_at_threshold": float((max_confidence(P["val_probs"]) >= thr).mean())}
    pred = P["test_probs"].argmax(1)
    p, r, f, s = precision_recall_fscore_support(te.label.values, pred, labels=range(len(classes)), zero_division=0)
    res["per_class"] = {c: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]), support=int(s[i])) for i, c in enumerate(classes)}
    res["confusion_matrix"] = confusion_matrix(te.label.values, pred, labels=range(len(classes))).tolist()
    # per-image table for galleries
    rows = []
    for sub, probs in ((te, P["test_probs"]), (unk, P["unknown_probs"])):
        for (_, r_), pr in zip(sub.iterrows(), probs):
            rows.append(dict(path=r_.path, true=r_.species, role=r_.role, pred=classes[pr.argmax()], conf=float(pr.max()), accepted=bool(pr.max() >= thr)))
    pd.DataFrame(rows).to_csv(MET / f"{key}_predictions.csv", index=False)
    return res, P


def fig_confusion(results, classes):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, (k, r) in zip(axes, results.items()):
        cm = np.array(r["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=45); ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{MODELS[k]} (known test, n={cm.sum()})")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrices.png", dpi=140); plt.close()


def fig_comparison(results):
    metrics = [("accuracy", "Known acc."), ("macro_f1", "Known macro-F1"), ("accuracy_on_accepted", "Acc. on accepted")]
    labels = [m[1] for m in metrics] + ["Unknown rejected", "AUROC"]
    fig, ax = plt.subplots(figsize=(9, 4)); w = 0.38
    for i, (k, r) in enumerate(results.items()):
        v = [r["known_test"][m[0]] for m in metrics] + [r["unknown"]["rejection_rate"], r["auroc_unknown_positive"]]
        b = ax.bar(np.arange(len(v)) + (i - .5) * w, v, w, label=MODELS[k], color=COLORS[k])
        ax.bar_label(b, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(labels)), labels); ax.set_ylim(0, 1.12); ax.legend(loc="lower left")
    ax.set_title("Model comparison on held-out test data (threshold fixed on validation)")
    plt.tight_layout(); plt.savefig(FIG / "model_comparison.png", dpi=140); plt.close()


def fig_confidence(results, probs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, k in zip(axes, results):
        ax.hist(max_confidence(probs[k]["test_probs"]), bins=30, range=(0, 1), alpha=.65, label="Known (test)", color="#2a9d5c")
        ax.hist(max_confidence(probs[k]["unknown_probs"]), bins=30, range=(0, 1), alpha=.65, label="Unknown species (test)", color="#d1495b")
        ax.axvline(results[k]["threshold"], color="k", ls="--", label=f"threshold {results[k]['threshold']:.3f}")
        ax.set_yscale("log"); ax.set_xlabel("Maximum class probability"); ax.set_ylabel("images (log)"); ax.set_title(MODELS[k]); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "confidence_distributions.png", dpi=140); plt.close()


def fig_risk_coverage(results, probs, df):
    te_y = df[df.split == "test"].label.values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k, r in results.items():
        conf, correct = combined_correctness(probs[k]["test_probs"], te_y, probs[k]["unknown_probs"])
        _, cov, risk = risk_coverage(conf, correct)
        axes[0].plot(cov, risk, color=COLORS[k], label=MODELS[k])
        axes[0].scatter([r["combined"]["coverage"]], [r["combined"]["risk"]], color=COLORS[k], zorder=3, marker="o", edgecolor="k")
        kc = max_confidence(probs[k]["test_probs"]); _, cov2, risk2 = risk_coverage(kc, probs[k]["test_probs"].argmax(1) == te_y)
        axes[1].plot(cov2, risk2, color=COLORS[k], label=MODELS[k])
        axes[1].scatter([r["known_test"]["coverage"]], [1 - r["known_test"]["accuracy_on_accepted"]], color=COLORS[k], zorder=3, edgecolor="k")
    axes[0].set_title("Known + unknown pool (unknown accepted = error)"); axes[1].set_title("Known test images only")
    for ax in axes:
        ax.set_xlabel("Coverage (fraction of images accepted)"); ax.set_ylabel("Risk (error rate among accepted)"); ax.legend(); ax.grid(alpha=.3)
    fig.text(.5, -.02, "Dots = operating point at the validation-chosen threshold", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "risk_coverage.png", dpi=140, bbox_inches="tight"); plt.close()


def fig_curves():
    h = pd.DataFrame(json.load(open("artifacts/logs/mobilenet_train.json"))["history"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="validation"); ax[0].set_title("Loss")
    ax[1].plot(h.epoch, h.train_acc, label="train"); ax[1].plot(h.epoch, h.val_acc, label="validation"); ax[1].set_title("Accuracy")
    for a in ax: a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(FIG / "mobilenet_training_curves.png", dpi=140); plt.close()


def fig_gallery(key, n=5, seed=42):
    """Seeded random sample per outcome category (not cherry-picked)."""
    d = pd.read_csv(MET / f"{key}_predictions.csv")
    known = d.role == "known"
    cats = {"Known, correct": d[known & d.accepted & (d.true == d.pred)],
            "Known, wrong species": d[known & d.accepted & (d.true != d.pred)],
            "Known, rejected": d[known & ~d.accepted],
            "Unknown, wrongly accepted": d[~known & d.accepted],
            "Unknown, rejected": d[~known & ~d.accepted]}
    fig, axes = plt.subplots(len(cats), n + 1, figsize=(2.1 * (n + 1), 2.4 * len(cats)),
                             gridspec_kw={"width_ratios": [0.9] + [1] * n})
    for r, (name, sub) in enumerate(cats.items()):
        axes[r, 0].axis("off")
        axes[r, 0].text(0.95, 0.5, f"{name}\n(total {len(sub)})", ha="right", va="center", fontsize=10, weight="bold")
        sub = sub.sample(min(n, len(sub)), random_state=seed)
        for c in range(n):
            ax = axes[r, c + 1]; ax.axis("off")
            if c < len(sub):
                x = sub.iloc[c]; ax.imshow(Image.open(x.path).convert("RGB"))
                ax.set_title(f"true: {x.true}\npred: {x.pred} ({x.conf:.2f})", fontsize=8)
            elif c == 0:
                ax.text(0.5, 0.5, "none occurred", ha="center", va="center", fontsize=9)
    plt.suptitle(f"{MODELS[key]} - random examples per outcome (confidence in brackets)", y=1.0)
    plt.tight_layout(); plt.savefig(FIG / f"failure_gallery_{key}.png", dpi=120, bbox_inches="tight"); plt.close()


def write_table(results):
    rows = []
    for k, r in results.items():
        kt, u, c = r["known_test"], r["unknown"], r["combined"]
        rows.append({"Model": MODELS[k], "Threshold": round(r["threshold"], 4), "Known acc": kt["accuracy"], "Known macro-F1": kt["macro_f1"],
                     "Known coverage": kt["coverage"], "False rejection": kt["false_rejection_rate"], "Acc. on accepted": kt["accuracy_on_accepted"],
                     "Unknown rejected": u["rejection_rate"], "Unknown accepted (FAR)": u["false_acceptance_rate"],
                     "AUROC (unknown=+)": r["auroc_unknown_positive"], "Pool coverage": c["coverage"], "Pool risk": c["risk"]})
    t = pd.DataFrame(rows)
    t.to_csv(MET / "results_table.csv", index=False)
    (MET / "results_table.md").write_text(t.to_markdown(index=False, floatfmt=".3f") if _has_tabulate() else t.round(3).to_string(index=False))
    return t


def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


def main():
    cfg = load_config(); df = load_metadata(); classes = list(cfg["known"])
    FIG.mkdir(parents=True, exist_ok=True); MET.mkdir(parents=True, exist_ok=True)
    results, probs = {}, {}
    for k in MODELS:
        results[k], probs[k] = evaluate_model(k, df, cfg)
    json.dump(results, open(MET / "open_set_results.json", "w"), indent=2)
    fig_confusion(results, classes); fig_comparison(results); fig_confidence(results, probs)
    fig_risk_coverage(results, probs, df); fig_curves()
    for k in MODELS:
        fig_gallery(k)
    print(write_table(results).round(3).T.to_string())
    for k, r in results.items():
        print(k, "per-unknown rejection:", {a: round(b, 3) for a, b in r["unknown_rejection_per_species"].items()},
              "AUROC:", {a: round(b, 3) for a, b in r["auroc_per_unknown_species"].items()}, "val acceptance", round(r["val_acceptance_at_threshold"], 3))


if __name__ == "__main__":
    main()
