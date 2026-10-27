"""HYBRID sensitivity split using the authors' leaf-count sheets (see src/leafmap.py).

* Images with an authoritative leaf ID (Apple, Peach, Tomato 'RS_HL' files): whole physical leaves are assigned to one split,
  so no matched leaf appears in more than one of train/val/test.
* Images WITHOUT an ID (Corn, Grape, Tomato 'GH_HL Leaf N' files): no authoritative grouping exists, so the earlier
  frame-block + guard-band HEURISTIC (src/grouped_split.py) is applied to them. These rows are labelled in `split_basis`.

The result is therefore NOT fully specimen-independent: for Corn, Grape and Tomato-GH (600 of 1,250 known images) the
separation rests on an unverified assumption about file numbering. Unknown-species images are untouched.
Reads data/metadata.csv, writes data/metadata_leafmap.csv."""
import argparse

import numpy as np
import pandas as pd

from src.data import load_config
from src.grouped_split import blockwise_split
from src.leafmap import annotate

SPLITS = ("train", "val", "test")


def allocate_groups(df, ratios, rng):
    """Assign whole groups (df.leaf_group) to splits by largest remaining deficit; returns a split Series."""
    target = {s: ratios[s] * len(df) for s in SPLITS}
    got = {s: 0 for s in SPLITS}
    groups = sorted(df.leaf_group.unique())
    rng.shuffle(groups)
    out = pd.Series(index=df.index, dtype=object)
    for g in groups:
        idx = df.index[df.leaf_group == g]
        s = max(SPLITS, key=lambda k: (target[k] - got[k], k == "train"))
        out.loc[idx] = s
        got[s] += len(idx)
    return out


def leafmap_split(df, ratios, seed, leaf_dir=None):
    ann = annotate(df) if leaf_dir is None else annotate(df, leaf_dir)
    out = ann.copy()
    out["split_basis"] = ""
    known = out.role == "known"
    rng = np.random.RandomState(seed)
    mapped = known & (out.leaf_status == "matched")
    for sp in sorted(out[mapped].species.unique()):
        sub = out[mapped & (out.species == sp)]
        out.loc[sub.index, "split"] = allocate_groups(sub, ratios, rng)
        out.loc[sub.index, "split_basis"] = "leaf_map"
    heur = known & ~mapped
    if heur.any():
        h = blockwise_split(out[heur].drop(columns=["leaf_status", "leaf_number", "leaf_group", "split_basis"]), ratios, seed)
        out.loc[h.index, "split"] = h["split"]
        out.loc[h.index, "split_basis"] = np.where(h.group_ns == "leaf", "filename_leaf_number_heuristic", "frame_block_heuristic")
        out.loc[h.index, "group_ns"], out.loc[h.index, "group_num"] = h.group_ns, h.group_num
    out["split_basis"] = out["split_basis"].replace("", "unknown_test")
    return out


def crossing_groups(df, split_col="split"):
    """Per species: authoritative leaf groups present in more than one of train/val/test, and test images sharing a leaf with train/val."""
    k = df[(df.role == "known") & (df.leaf_status == "matched") & df[split_col].isin(SPLITS)]
    rows = []
    for sp, g in k.groupby("species"):
        n = g.groupby("leaf_group")[split_col].nunique()
        cross = n[n > 1].index
        t = g[g[split_col] == "test"]
        rows.append(dict(species=sp, images=len(g), leaves=int(g.leaf_group.nunique()), leaves_crossing=int(len(cross)),
                         images_in_crossing_leaves=int(g.leaf_group.isin(cross).sum()),
                         test_images=len(t), test_images_sharing_leaf_with_train_or_val=int(t.leaf_group.isin(g[g[split_col] != "test"].leaf_group).sum())))
    return pd.DataFrame(rows, columns=["species", "images", "leaves", "leaves_crossing", "images_in_crossing_leaves",
                                       "test_images", "test_images_sharing_leaf_with_train_or_val"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/metadata.csv")
    ap.add_argument("--out", default="data/metadata_leafmap.csv")
    a = ap.parse_args()
    cfg = load_config()
    out = leafmap_split(pd.read_csv(a.src), cfg["split"], cfg["seed"])
    out.to_csv(a.out, index=False)
    print(out.groupby(["species", "split_basis", "split"]).size().unstack(fill_value=0))
    print(crossing_groups(out).to_string(index=False))


if __name__ == "__main__":
    main()
