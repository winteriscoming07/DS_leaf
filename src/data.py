"""Dataset discovery, integrity checks, deduplication and leakage-free splits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_config(path="configs/experiment.json"):
    with open(path) as f:
        return json.load(f)


def discover_images(folder):
    """Sorted image files in a class folder; raises if the folder is missing."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Class folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def check_image(path):
    """Return (ok, width, height, mode). Corrupt/unreadable files give ok=False."""
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            w, h = im.size
            return True, w, h, im.mode
    except Exception:
        return False, 0, 0, ""


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_split(df, ratios, seed):
    """Add a 'split' column (train/val/test) stratified by species. Deterministic."""
    train_r, val_r, test_r = ratios["train"], ratios["val"], ratios["test"]
    assert abs(train_r + val_r + test_r - 1) < 1e-9
    idx = df.index.to_numpy()
    tr, rest = train_test_split(idx, train_size=train_r, stratify=df.loc[idx, "species"], random_state=seed)
    va, te = train_test_split(rest, train_size=val_r / (val_r + test_r), stratify=df.loc[rest, "species"], random_state=seed)
    out = df.copy()
    out.loc[tr, "split"], out.loc[va, "split"], out.loc[te, "split"] = "train", "val", "test"
    return out


def build_metadata(cfg):
    """Build the full metadata table. Order: discover -> drop corrupt -> drop exact
    duplicates (global, by hash) -> deterministic cap -> stratified split of known classes."""
    rng = np.random.RandomState(cfg["seed"])
    raw = Path(cfg["raw_dir"])
    rows, log = [], {"corrupt": [], "exact_duplicates_removed": [], "raw_counts": {}}
    seen = {}  # sha256 -> first path kept (global, so duplicates cannot cross roles/splits)
    for role, mapping, cap in (("known", cfg["known"], cfg["max_per_known"]),
                               ("unknown", cfg["unknown"], cfg["max_per_unknown"])):
        for species, folder in mapping.items():
            files = discover_images(raw / folder)
            log["raw_counts"][species] = len(files)
            kept = []
            for p in files:
                ok, w, h, mode = check_image(p)
                if not ok:
                    log["corrupt"].append(str(p))
                    continue
                digest = file_sha256(p)
                if digest in seen:
                    log["exact_duplicates_removed"].append([str(p), seen[digest]])
                    continue
                seen[digest] = str(p)
                kept.append(dict(path=str(p), source_class=folder, species=species, role=role,
                                 width=w, height=h, mode=mode, sha256=digest))
            if len(kept) > cap:
                pick = sorted(rng.choice(len(kept), cap, replace=False))
                kept = [kept[i] for i in pick]
            rows.extend(kept)
    df = pd.DataFrame(rows)
    df["split"] = "unknown_test"
    known = df[df.role == "known"]
    df.loc[known.index, "split"] = stratified_split(known, cfg["split"], cfg["seed"])["split"]
    df["label"] = -1
    classes = list(cfg["known"].keys())
    for i, c in enumerate(classes):
        df.loc[df.species == c, "label"] = i
    df["flag_non_rgb"] = df["mode"] != "RGB"
    return df, log


def load_metadata(path="data/metadata.csv"):
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.json")
    ap.add_argument("--out", default="data/metadata.csv")
    a = ap.parse_args()
    cfg = load_config(a.config)
    df, log = build_metadata(cfg)
    df.to_csv(a.out, index=False)
    log["class_to_label"] = {c: i for i, c in enumerate(cfg["known"])}
    log["source_folder_map"] = {**cfg["known"], **cfg["unknown"]}
    log["counts"] = df.groupby(["role", "species", "split"]).size().rename("n").reset_index().to_dict("records")
    Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
    with open("artifacts/metrics/data_build_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(df.groupby(["role", "species", "split"]).size())
    print("corrupt:", len(log["corrupt"]), "exact duplicates removed:", len(log["exact_duplicates_removed"]))


if __name__ == "__main__":
    main()
