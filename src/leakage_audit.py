"""Leakage audit: what specimen-level information exists, how close are test frames to train frames, and how do the results
change under the frame-block sensitivity split? Writes artifacts/metrics/leakage_audit.json, split_sensitivity.md and
artifacts/figures/frame_neighbours.png. Requires the primary and the LEAF_VARIANT=blockwise pipeline outputs."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.grouped_split import parse_name

MET, FIG = "artifacts/metrics", "artifacts/figures"


def parsed(df):
    p = df.path.map(parse_name)
    return df.assign(ns=[x[0] for x in p], num=[x[1] for x in p])


def proximity(df):
    """For each val/test image with a frame number: gap to the nearest train image of the same species + prefix."""
    k = parsed(df[df.role == "known"]); k = k[k.ns != "leaf"]
    rows = []
    for (sp, ns), g in k.groupby(["species", "ns"]):
        tr = g[g.split == "train"].num.values
        for s in ("val", "test"):
            d = np.array([np.abs(tr - x).min() for x in g[g.split == s].num.values])
            if d.size:
                rows.append(dict(species=sp, split=s, n=int(d.size), within_1=int((d <= 1).sum()), within_5=int((d <= 5).sum()), median_gap=float(np.median(d))))
    return rows


def montage(df, out, per_species=3, seed=1):
    k = parsed(df[(df.role == "known")]); k = k[k.ns != "leaf"]
    sps = list(dict.fromkeys(k.species))
    fig, axes = plt.subplots(len(sps), 2 * per_species, figsize=(13, 2.3 * len(sps)))
    for r, sp in enumerate(sps):
        g = k[k.species == sp]; tr = g[g.split == "train"]
        te = g[g.split == "test"].sample(per_species, random_state=seed)  # seeded random pairs, not cherry-picked
        for c, (_, t) in enumerate(te.iterrows()):
            n = tr.iloc[(tr.num - t.num).abs().argmin()]
            for j, (x, lab) in enumerate(((t, "TEST"), (n, "nearest-frame TRAIN"))):
                ax = axes[r, 2 * c + j]; ax.imshow(Image.open(x.path).convert("RGB")); ax.axis("off")
                ax.set_title(f"{sp} {lab}\nframe {x.num}", fontsize=7)
    plt.suptitle("Random test images and their nearest-numbered training image (same species and file prefix)", fontsize=10)
    plt.tight_layout(); plt.savefig(out, dpi=90); plt.close()


def to_md(df):
    cells = [[str(c) for c in df.columns]] + [[str(v) for v in r] for r in df.itertuples(index=False)]
    line = lambda r: "| " + " | ".join(r) + " |"
    return "\n".join([line(cells[0]), line(["---"] * len(cells[0]))] + [line(r) for r in cells[1:]]) + "\n"


def sensitivity_table(a, b, ca, cb):
    rows = []
    for k, name in (("hog_svm", "HOG + SVM"), ("mobilenet", "MobileNetV3-Small")):
        for tag, r, ci in (("primary (image-level)", a[k], ca[k]), ("frame-block + guard", b[k], cb[k])):
            kt = r["known_test"]
            rows.append({"Model": name, "Split": tag, "Known test n": kt["n"], "Known acc": kt["accuracy"], "Threshold": r["threshold"],
                         "Unknown rejected": r["unknown"]["rejection_rate"],
                         "Potato rejected": r["unknown_rejection_per_species"]["Potato"], "Bell pepper rejected": r["unknown_rejection_per_species"]["Bell pepper"],
                         "AUROC": r["auroc_unknown_positive"], "AUROC 95% CI": f"{ci['auroc'][0]:.3f}-{ci['auroc'][1]:.3f}"})
    return pd.DataFrame(rows)


def main():
    prim, blk = pd.read_csv("data/metadata.csv"), pd.read_csv("data/metadata_blockwise.csv")
    leaf = parsed(prim[prim.role == "known"]); leaf = leaf[leaf.ns == "leaf"]
    span = leaf.groupby("num").split.nunique()
    rep = {"known_images": int((prim.role == "known").sum()),
           "images_with_explicit_leaf_number": int(len(leaf)), "distinct_leaf_numbers": int(leaf.num.nunique()),
           "leaf_numbers_spanning_multiple_splits_in_primary": int((span > 1).sum()),
           "images_whose_leaf_number_appears_in_another_split": int(leaf.num.isin(span[span > 1].index).sum()),
           "frame_proximity_primary": proximity(prim),
           "blockwise_split_counts": blk.groupby(["species", "split"]).size().unstack(fill_value=0).to_dict("index"),
           "note": "Frame numbers are camera sequence numbers; adjacency does NOT prove the same physical leaf. Leaf numbers exist for Tomato 'GH_HL Leaf' files only."}
    json.dump(rep, open(f"{MET}/leakage_audit.json", "w"), indent=2)
    montage(prim, f"{FIG}/frame_neighbours.png")
    a, b = (json.load(open(f"{p}/metrics/open_set_results.json")) for p in ("artifacts", "artifacts/variant_blockwise"))
    ca, cb = (json.load(open(f"{p}/metrics/bootstrap_ci.json")) for p in ("artifacts", "artifacts/variant_blockwise"))
    t = sensitivity_table(a, b, ca, cb)
    t.to_csv(f"{MET}/split_sensitivity.csv", index=False)
    open(f"{MET}/split_sensitivity.md", "w").write(to_md(t.round(3)))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("frame_proximity_primary", "blockwise_split_counts")}, indent=1))
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
