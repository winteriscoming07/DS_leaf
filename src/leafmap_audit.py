"""Leaf-metadata audit: coverage of authoritative leaf IDs for the 1,550 selected images, leaf crossings in every split manifest,
statistical-independence consequences for the unknown pools, and a leaf-clustered bootstrap.
Writes artifacts/metrics/leafmap_audit.json and leafmap_audit.md. Requires the primary, blockwise and leafmap pipeline outputs.

Cluster bootstrap: resample whole physical leaves (images without an authoritative ID are singleton clusters). It removes the
independence assumption for leaves that have an ID but still holds model, threshold and split fixed."""
import json

import numpy as np
import pandas as pd

from src.leafmap import annotate, check_provenance
from src.leafmap_split import crossing_groups
from src.open_set import auroc_unknown_vs_known, max_confidence

VARIANTS = {"primary (image-level)": ("data/metadata.csv", "artifacts"),
            "frame-block heuristic": ("data/metadata_blockwise.csv", "artifacts/variant_blockwise"),
            "hybrid leaf-map": ("data/metadata_leafmap.csv", "artifacts/variant_leafmap")}


def cluster_ids(sub):
    """Cluster label per image: authoritative leaf group if matched, else the image itself."""
    return np.where(sub.leaf_status == "matched", sub.leaf_group.fillna(""), "single:" + sub.path)


def cluster_bootstrap(manifest, art, model, n_boot=2000, seed=42):
    df = annotate(pd.read_csv(manifest)) if "leaf_status" not in pd.read_csv(manifest).columns else pd.read_csv(manifest)
    te, unk = df[df.split == "test"], df[df.split == "unknown_test"]
    P = np.load(f"{art}/predictions/{model}.npz"); thr = json.load(open(f"{art}/metrics/open_set_results.json"))[model]["threshold"]
    kp, up = P["test_probs"], P["unknown_probs"]; kc, uc = max_confidence(kp), max_confidence(up)
    kcorrect = kp.argmax(1) == te.label.values
    kg, ug = pd.Series(range(len(te))).groupby(cluster_ids(te)).apply(list).tolist(), pd.Series(range(len(unk))).groupby(cluster_ids(unk)).apply(list).tolist()
    rng = np.random.RandomState(seed); vals = {m: [] for m in ("known_accuracy", "unknown_rejection", "auroc")}
    for _ in range(n_boot):
        i = np.concatenate([kg[j] for j in rng.randint(0, len(kg), len(kg))]); j = np.concatenate([ug[k] for k in rng.randint(0, len(ug), len(ug))])
        vals["known_accuracy"].append(kcorrect[i].mean()); vals["unknown_rejection"].append((uc[j] < thr).mean())
        vals["auroc"].append(auroc_unknown_vs_known(kc[i], uc[j]))
    out = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in vals.items()}
    out["n_known_test_clusters"], out["n_unknown_clusters"] = len(kg), len(ug)
    return out


def main():
    prov_bad = check_provenance()
    prim = annotate(pd.read_csv("data/metadata.csv"))
    cov = prim.groupby(["role", "species"]).apply(lambda g: pd.Series({
        "images": len(g), "matched_authoritative_id": int((g.leaf_status == "matched").sum()),
        "pct_matched": round(100 * (g.leaf_status == "matched").mean(), 1),
        "no_sheet_upstream": int((g.leaf_status == "no_sheet").sum()), "not_in_sheet": int((g.leaf_status == "not_in_sheet").sum()),
        "ambiguous": int((g.leaf_status == "ambiguous").sum()), "distinct_leaves": int(g.leaf_group.nunique())}), include_groups=False).reset_index()
    unknown = {}
    for sp, g in prim[prim.role == "unknown"].groupby("species"):
        s = g.groupby("leaf_group").size()
        unknown[sp] = {"images": len(g), "distinct_leaves": int(len(s)), "images_per_leaf_median": float(s.median()), "images_per_leaf_max": int(s.max()),
                       "design_effect_note": "images from one leaf are correlated; effective sample size is between #leaves and #images"}
    rep = {"provenance_mismatches": prov_bad, "coverage": cov.to_dict("records"), "unknown_independence": unknown, "crossings": {}, "cluster_bootstrap": {}}
    for name, (manifest, art) in VARIANTS.items():
        d = annotate(pd.read_csv(manifest))
        rep["crossings"][name] = crossing_groups(d).to_dict("records")
        rep["cluster_bootstrap"][name] = {m: cluster_bootstrap(manifest, art, m) for m in ("hog_svm", "mobilenet")}
    json.dump(rep, open("artifacts/metrics/leafmap_audit.json", "w"), indent=2)
    lines = ["## Coverage of authoritative leaf IDs", pd.DataFrame(rep["coverage"]).to_string(index=False), "", "## Leaves crossing train/val/test (matched images only)"]
    for name, rows in rep["crossings"].items():
        lines += [f"### {name}", pd.DataFrame(rows).to_string(index=False)]
    lines += ["", "## Leaf-clustered 95% bootstrap intervals (2000 resamples)"]
    for name, d in rep["cluster_bootstrap"].items():
        for m, v in d.items():
            lines.append(f"{name:24s} {m:10s} " + "  ".join(f"{k}={v[k][0]:.3f}-{v[k][1]:.3f}" for k in ("known_accuracy", "unknown_rejection", "auroc"))
                         + f"  (clusters: known {v['n_known_test_clusters']}, unknown {v['n_unknown_clusters']})")
    open("artifacts/metrics/leafmap_audit.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("\nunknown independence:", json.dumps(unknown, indent=1))


if __name__ == "__main__":
    main()
