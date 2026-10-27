"""Dataset inspection report: counts, sizes, channels, near-duplicates, montage."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data import load_config, load_metadata


def ahash(path, n=8):
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)
    return (g > g.mean()).flatten()


def near_duplicates(df, max_dist=2):
    H = np.stack([ahash(p) for p in df.path])
    d = (H[:, None, :] != H[None, :, :]).sum(-1)
    i, j = np.where(np.triu(d <= max_dist, k=1))
    return [(int(a), int(b)) for a, b in zip(i, j)]


def main():
    cfg = load_config()
    df = load_metadata()
    Path("artifacts/figures").mkdir(parents=True, exist_ok=True)
    counts = df.groupby(["role", "species"]).size()
    known_counts = counts["known"]
    rep = {
        "total_images": int(len(df)),
        "class_counts": {f"{r}/{s}": int(n) for (r, s), n in counts.items()},
        "known_imbalance_ratio_max_over_min": float(known_counts.max() / known_counts.min()),
        "width": df.width.describe().to_dict(), "height": df.height.describe().to_dict(),
        "unique_sizes": {f"{w}x{h}": int(n) for (w, h), n in df.groupby(["width", "height"]).size().items()},
        "color_modes": df["mode"].value_counts().to_dict(),
    }
    # per-channel mean/std on a deterministic sample of known-train images
    sample = df[df.split == "train"].sample(100, random_state=cfg["seed"])
    px = np.concatenate([np.asarray(Image.open(p).convert("RGB").resize((64, 64)), dtype=np.float32).reshape(-1, 3) / 255 for p in sample.path])
    rep["train_rgb_mean"], rep["train_rgb_std"] = px.mean(0).round(4).tolist(), px.std(0).round(4).tolist()
    nd = {}
    for d in (0, 1, 2):
        pr = near_duplicates(df, d)
        same = [(a, b) for a, b in pr if df.species[a] == df.species[b]]
        nd[f"aHash_dist<={d}"] = {"pairs": len(pr), "same_species": len(same),
                                  "same_species_cross_split": sum(df.split[a] != df.split[b] for a, b in same),
                                  "cross_species": len(pr) - len(same)}
    rep["near_duplicates_ahash8x8"] = nd
    rep["near_duplicate_note"] = ("8x8 average hash is coarse; flagged same-species cross-split pairs were visually "
                                  "inspected and are different leaves with similar shape/background (false positives). "
                                  "Authoritative leaf IDs exist for Apple, Peach, Tomato (RS_HL only), Potato and Bell pepper (see leafmap_audit.json) and show many same-leaf photos across splits; Corn, Grape and Tomato GH_HL have no leaf IDs upstream.")
    build = json.load(open("artifacts/metrics/data_build_log.json"))
    rep["corrupt_files"] = len(build["corrupt"])
    rep["exact_duplicates_removed"] = len(build["exact_duplicates_removed"])
    rep["class_to_label"] = build["class_to_label"]
    rep["source_folder_map"] = build["source_folder_map"]
    json.dump(rep, open("artifacts/metrics/dataset_report.json", "w"), indent=2)

    species = list(cfg["known"]) + list(cfg["unknown"])
    fig, axes = plt.subplots(len(species), 6, figsize=(10, 1.7 * len(species)))
    for r, s in enumerate(species):
        paths = df[df.species == s].sample(6, random_state=cfg["seed"]).path
        for c, p in enumerate(paths):
            axes[r, c].imshow(Image.open(p).convert("RGB")); axes[r, c].axis("off")
        role = "UNKNOWN" if s in cfg["unknown"] else "known"
        axes[r, 0].set_title(f"{s} ({role})", loc="left", fontsize=9)
    plt.tight_layout(); plt.savefig("artifacts/figures/sample_montage.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    piv = df.groupby(["species", "split"]).size().unstack(fill_value=0)
    piv.plot.bar(stacked=True, ax=ax[0]); ax[0].set_title("Images per species and split"); ax[0].set_ylabel("images")
    ax[1].hist(df.width, bins=20, alpha=.6, label="width"); ax[1].hist(df.height, bins=20, alpha=.6, label="height")
    ax[1].set_title("Image dimensions (px)"); ax[1].legend()
    plt.tight_layout(); plt.savefig("artifacts/figures/dataset_overview.png", dpi=130); plt.close()
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
