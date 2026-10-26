"""SENSITIVITY split: frame-block grouping. NOT a specimen-level split.

PlantVillage does not publish physical-leaf IDs. The only identifiers in the file names are
  * 'GH_HL Leaf <N>[.<k>]'  (100 Tomato images): N looks like a leaf number, used here as a group key;
  * camera frame numbers ('RS_HL 5672', 'Mt.N.V_HL 6067', ...): consecutive frames MAY show the same leaf, but this is
    an assumption, not a documented fact.
This module keeps frame-adjacent images together (blocks of `block` frames) and then drops images that are within
`guard` frames of an image in a different split, so every cross-split pair is >= guard+1 frames apart. It is a
robustness check against the *possibility* of near-same-leaf overlap; it cannot prove specimen independence.

Reads data/metadata.csv (same images, same unknown set) and writes data/metadata_blockwise.csv."""
import argparse
import re

import numpy as np
import pandas as pd

from src.data import load_config

BLOCK, GUARD = 50, 5
_LEAF = re.compile(r"_HL Leaf (\d+)")
_FRAME = re.compile(r"_HL (\d+)")


def parse_name(path):
    """-> (namespace, number). namespace 'leaf' = explicit leaf number in the name, else 'frame:<prefix>'."""
    name = str(path).replace("\\", "/").split("/")[-1]
    stem = re.sub(r"^[0-9a-f-]{36}___", "", name)
    m = _LEAF.search(stem)
    if m:
        return "leaf", int(m.group(1))
    m = _FRAME.search(stem)
    if not m:
        raise ValueError(f"cannot parse frame/leaf number from {name!r}")
    return "frame:" + stem[:m.start()], int(m.group(1))


def blockwise_split(df, ratios, seed, block=BLOCK, guard=GUARD):
    """Return a copy of df with known-species rows re-split by group. Dropped rows get split='guard_excluded'."""
    out = df.copy()
    parsed = out.path.map(parse_name)
    out["group_ns"], out["group_num"] = [p[0] for p in parsed], [p[1] for p in parsed]
    is_leaf = out.group_ns == "leaf"
    out["group"] = np.where(is_leaf, "leaf-" + out.group_num.astype(str),
                            out.species + "|" + out.group_ns + "|b" + (out.group_num // block).astype(str))
    rng = np.random.RandomState(seed)
    known = out.role == "known"
    for sp in sorted(out[known].species.unique()):
        idx = out[known & (out.species == sp)]
        groups = sorted(idx.group.unique())
        rng.shuffle(groups)
        target = {s: ratios[s] * len(idx) for s in ("train", "val", "test")}
        got = {s: 0 for s in target}
        for g in groups:
            rows = idx.index[idx.group == g]
            s = max(target, key=lambda k: (target[k] - got[k], k == "train"))  # biggest remaining deficit
            out.loc[rows, "split"] = s
            got[s] += len(rows)
    # guard band: keep TEST intact, drop val near test, then drop train near val/test (same species + namespace only)
    for (sp, ns), g in out[known & (out.group_ns != "leaf")].groupby(["species", "group_ns"]):
        for drop, ref in (("val", ["test"]), ("train", ["val", "test"])):
            r = g[g.split.isin(ref)].group_num.values
            for i in g.index[(g.split == drop)]:
                if r.size and np.abs(r - g.group_num[i]).min() <= guard:
                    out.loc[i, "split"] = "guard_excluded"
    return out


def min_cross_split_gap(df, a, b):
    """Smallest |frame gap| between any image in split a and any in split b (same species + namespace). inf if none."""
    best = np.inf
    k = df[(df.role == "known") & (df.group_ns != "leaf")]
    for _, g in k.groupby(["species", "group_ns"]):
        x, y = g[g.split == a].group_num.values, g[g.split == b].group_num.values
        if x.size and y.size:
            best = min(best, np.abs(x[:, None] - y[None, :]).min())
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/metadata.csv")
    ap.add_argument("--out", default="data/metadata_blockwise.csv")
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--guard", type=int, default=GUARD)
    a = ap.parse_args()
    cfg = load_config()
    df = pd.read_csv(a.src)
    out = blockwise_split(df, cfg["split"], cfg["seed"], a.block, a.guard)
    out.to_csv(a.out, index=False)
    print(out.groupby(["species", "split"]).size().unstack(fill_value=0))
    for pair in (("train", "val"), ("train", "test"), ("val", "test")):
        print("min frame gap", pair, min_cross_split_gap(out, *pair))
    lk = out[out.group_ns == "leaf"]
    print("GH leaf ids spanning >1 split:", int((lk.groupby("group_num").split.nunique() > 1).sum()))


if __name__ == "__main__":
    main()
