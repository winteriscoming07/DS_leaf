"""Authoritative physical-leaf IDs from the PlantVillage authors' leaf-count sheets (preserved in data/leaf_maps/, see PROVENANCE.json).

Each sheet row maps an ORIGINAL file name ('RS_HL 7513.JPG') to the 'Leaf #' counted by the dataset authors for that class.
Nothing is inferred: an image is 'matched' only when its normalised name appears exactly once in the species' own sheet.
Corn and Grape (healthy) have no sheet upstream, so their images are never given a leaf ID here.

Group keys are '<species>|<sheet file>|<leaf number>': leaf numbers restart per sheet and are not unique across classes."""
import csv
import json
import re
from pathlib import Path

LEAF_DIR = Path("data/leaf_maps")
SHEETS = {"Apple": "Apple_Healthy.csv", "Peach": "Peach_Healthy.csv", "Tomato": "Tomato_Healthy.csv",
          "Bell pepper": "Pepper_Healthy.csv", "Potato": "Potato_Healthy Potato.csv"}
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}___")


def normalize_name(path_or_name):
    """'dir/<uuid>___RS_HL 7513.JPG' -> 'rs_hl 7513'. Strips the UUID prefix, a trailing ' copy[ n]' marker and the extension."""
    name = str(path_or_name).replace("\\", "/").split("/")[-1]
    name = _UUID.sub("", name)
    name = re.sub(r"\.(jpe?g|png)$", "", name, flags=re.I)
    name = re.sub(r"\s*copy(\s*\d+)?$", "", name, flags=re.I)
    return name.strip().lower()


def load_sheet(path):
    """-> {normalised name: set of leaf numbers (as ints)}. A name with >1 distinct leaf number is ambiguous."""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(normalize_name(r["File Name"]), set()).add(int(float(r["Leaf #"])))
    return out


def load_sheets(leaf_dir=LEAF_DIR):
    return {sp: load_sheet(Path(leaf_dir) / f) for sp, f in SHEETS.items() if (Path(leaf_dir) / f).exists()}


def lookup(species, path, sheets):
    """-> (status, leaf, group). status: matched | no_sheet | not_in_sheet | ambiguous."""
    if species not in SHEETS or species not in sheets:
        return "no_sheet", None, None
    leaves = sheets[species].get(normalize_name(path))
    if not leaves:
        return "not_in_sheet", None, None
    if len(leaves) > 1:
        return "ambiguous", None, None
    leaf = next(iter(leaves))
    return "matched", leaf, f"{species}|{SHEETS[species]}|{leaf}"


def annotate(df, leaf_dir=LEAF_DIR):
    """Add leaf_status / leaf_number / leaf_group columns. Reads only the species' own sheet."""
    sheets = load_sheets(leaf_dir)
    res = [lookup(sp, p, sheets) for sp, p in zip(df.species, df.path)]
    out = df.copy()
    out["leaf_status"], out["leaf_number"], out["leaf_group"] = ([r[i] for r in res] for i in range(3))
    return out


def check_provenance(leaf_dir=LEAF_DIR):
    """Verify the preserved sheets against the sha256 values recorded in PROVENANCE.json. Returns list of mismatches."""
    import hashlib
    prov = json.load(open(Path(leaf_dir) / "PROVENANCE.json"))
    return [n for n, m in prov["files"].items() if hashlib.sha256((Path(leaf_dir) / n).read_bytes()).hexdigest() != m["sha256"]]
