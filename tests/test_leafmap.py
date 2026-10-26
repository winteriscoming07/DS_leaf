from pathlib import Path

import uuid

import pandas as pd
import pytest

from src.leafmap import SHEETS, annotate, check_provenance, load_sheet, load_sheets, lookup, normalize_name
from src.leafmap_split import crossing_groups, leafmap_split

UUID = "0055dd26-23a7-4415-ac61-e0b44ebfaf80"
RATIOS = {"train": .7, "val": .15, "test": .15}
needs_sheets = pytest.mark.skipif(not Path("data/leaf_maps/PROVENANCE.json").exists(), reason="leaf maps not present")


def test_normalize_name_strips_uuid_extension_copy_and_case():
    assert normalize_name(f"data/raw/Apple___healthy/{UUID}___RS_HL 7513.JPG") == "rs_hl 7513"
    assert normalize_name("RS_HL 7513.JPG") == "rs_hl 7513"
    assert normalize_name(f"{UUID}___R.S_HL 8315 copy 2.jpg") == "r.s_hl 8315"
    assert normalize_name(f"{UUID}___R.S_HL 8189 copy.jpg") == "r.s_hl 8189"
    assert normalize_name(f"{UUID}___GH_HL Leaf 259.1.JPG") == "gh_hl leaf 259.1"


@needs_sheets
def test_provenance_checksums_match_preserved_files():
    assert check_provenance() == []


@needs_sheets
def test_upstream_examples_map_to_expected_leaves():
    s = load_sheets()
    assert lookup("Apple", f"{UUID}___RS_HL 7513.JPG", s) == ("matched", 165, "Apple|Apple_Healthy.csv|165")
    assert lookup("Peach", f"{UUID}___Rutg._HL 3671.JPG", s)[1] == 46
    assert lookup("Tomato", f"{UUID}___RS_HL 0218.JPG", s)[1] == 54


@needs_sheets
def test_missing_ids_are_never_invented():
    s = load_sheets()
    assert lookup("Corn", f"{UUID}___R.S_HL 8315 copy 2.jpg", s)[0] == "no_sheet"       # no Corn sheet upstream
    assert lookup("Grape", f"{UUID}___Mt.N.V_HL 6067.JPG", s)[0] == "no_sheet"          # no Grape-healthy sheet upstream
    assert lookup("Tomato", f"{UUID}___GH_HL Leaf 259.1.JPG", s) == ("not_in_sheet", None, None)  # filename number is NOT an authoritative ID
    assert lookup("Apple", f"{UUID}___RS_HL 99999.JPG", s)[0] == "not_in_sheet"


def test_ambiguous_names_are_flagged_not_resolved(tmp_path):
    p = tmp_path / "Apple_Healthy.csv"
    p.write_text('"File Name","Leaf #"\n"RS_HL 1.JPG","1.0"\n"RS_HL 1.JPG","2.0"\n"RS_HL 2.JPG","3.0"\n')
    s = {"Apple": load_sheet(p)}
    assert lookup("Apple", "RS_HL 1.JPG", s)[0] == "ambiguous"
    assert lookup("Apple", "RS_HL 2.JPG", s)[:2] == ("matched", 3)


def test_group_keys_are_scoped_by_species_and_sheet(tmp_path):
    for name in ("Apple_Healthy.csv", "Peach_Healthy.csv"):
        (tmp_path / name).write_text('"File Name","Leaf #"\n"X 1.JPG","7.0"\n')
    df = pd.DataFrame({"species": ["Apple", "Peach"], "path": ["a/X 1.JPG", "b/X 1.JPG"]})
    a = annotate(df, tmp_path)
    assert a.leaf_number.tolist() == [7, 7] and a.leaf_group.nunique() == 2


def _u(i):
    return str(uuid.UUID(int=i))


def _toy(tmp_path, n_leaves=40, per=5):
    sheet = ['"File Name","Leaf #"']; rows = []
    for leaf in range(n_leaves):
        for j in range(per):
            fr = leaf * per + j; sheet.append(f'"RS_HL {fr}.JPG","{leaf + 1}.0"')
            rows.append(dict(path=f"d/{_u(fr)}___RS_HL {fr}.JPG", species="Apple", role="known", split="train", label=0))
    for i in range(60):  # no sheet -> heuristic
        rows.append(dict(path=f"d/{_u(1000 + i)}___Mt.N.V_HL {5000 + 3 * i}.JPG", species="Grape", role="known", split="train", label=2))
    rows += [dict(path=f"u/{_u(2000 + i)}___JR_HL {i}.JPG", species="Bell pepper", role="unknown", split="unknown_test", label=-1) for i in range(10)]
    (tmp_path / "Apple_Healthy.csv").write_text("\n".join(sheet) + "\n")
    return pd.DataFrame(rows)


def test_leaf_groups_never_cross_splits_and_heuristic_rows_are_labelled(tmp_path):
    out = leafmap_split(_toy(tmp_path), RATIOS, 42, tmp_path)
    cg = crossing_groups(out)
    assert len(cg) == 1 and cg.leaves.iloc[0] == 40 and cg.leaves_crossing.iloc[0] == 0
    ap = out[out.species == "Apple"]
    assert (ap.split_basis == "leaf_map").all() and set(ap.split) == {"train", "val", "test"}
    gr = out[out.species == "Grape"]
    assert (gr.split_basis == "frame_block_heuristic").all()
    assert (out[out.role == "unknown"].split == "unknown_test").all()
    assert (out[out.role == "unknown"].split_basis == "unknown_test").all()


def test_leafmap_split_is_reproducible_and_seed_sensitive(tmp_path):
    df = _toy(tmp_path)
    a, b = leafmap_split(df, RATIOS, 42, tmp_path), leafmap_split(df, RATIOS, 42, tmp_path)
    pd.testing.assert_frame_equal(a, b)
    assert not a.split.equals(leafmap_split(df, RATIOS, 7, tmp_path).split)


@needs_sheets
@pytest.mark.skipif(not Path("data/metadata_leafmap.csv").exists(), reason="run src.leafmap_split first")
def test_committed_leafmap_manifest_invariants():
    d = pd.read_csv("data/metadata_leafmap.csv"); base = pd.read_csv("data/metadata.csv")
    assert set(d.path) == set(base.path) and d.path.is_unique
    assert crossing_groups(d).leaves_crossing.sum() == 0
    assert (d[d.role == "unknown"].split == "unknown_test").all()
    cov = d[d.role == "known"].groupby("species").leaf_status.apply(lambda s: (s == "matched").mean()).to_dict()
    assert cov["Apple"] == cov["Peach"] == 1.0 and cov["Corn"] == cov["Grape"] == 0.0 and abs(cov["Tomato"] - .6) < 1e-9
    assert not (d[d.leaf_status != "matched"].split_basis == "leaf_map").any()
