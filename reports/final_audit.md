# Final audit (2026-10-27, corrected after inspecting the upstream leaf-grouping metadata)

Scope: scientific validity first, then completeness. Existing work was preserved; primary results were **audited, not regenerated** (see §6). Everything below was run in this repository; nothing is quoted from memory.

## 1. Baseline
- Git: clean tree at `7d8d1ee`; 8 commits. Tests: **13 passed** (Python 3.11.15). Primary artifacts present (`data/metadata.csv`, predictions, checkpoints, metrics, figures).
- Accidental file found and removed: `nothing-0.0.3-py2.py3-none-any.whl` (tracked, a trivial PyPI package, referenced nowhere). Recoverable from history (`git show 7d8d1ee:nothing-0.0.3-py2.py3-none-any.whl`).

## 2. Data leakage investigation

> **Correction (2026-10-27).** The first version of this section said PlantVillage has no physical-leaf IDs and that a specimen-level split was impossible. That was wrong: only file names had been inspected and the upstream repository had not been checked for grouping metadata. The upstream repository contains the dataset authors' leaf-count sheets. This section was rewritten after inspecting them at the pinned commit.

**Question:** can different photographs of the same physical leaf occur in train and test?

**Finding: yes, it happens, extensively, in the primary split. A fully group-aware split is not possible for all five known classes because two have no upstream mapping.**

**Source.** `spMohanty/PlantVillage-Dataset` @ `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`, `leaf_grouping/leaf_maps/*.csv` (columns `File Name`, `Leaf #`; counted by the dataset authors). The five relevant sheets are preserved unmodified in `data/leaf_maps/` with git blob SHAs and SHA-256 checksums in `PROVENANCE.json` (verified by a test). They are byte-identical to the copies under `filtered_leafmaps/`. The pinned tree contains no other grouping source: `leaf-map.json` has no Corn or Grape-healthy entries, and the Grape workbook holds only the three disease sheets. Images are matched by normalising the file name (strip UUID prefix, extension, `copy` marker, lowercase) against the **species' own sheet only**; group keys are `<species>|<sheet>|<leaf>` because leaf numbers restart per sheet (`src/leafmap.py`). Nothing is inferred; 0 ambiguous matches occurred.

**Coverage of the 1,550 selected images** (`artifacts/metrics/leafmap_audit.md`):

| Species | Role | Images | Authoritative leaf ID | Not matched | Ambiguous | Distinct leaves |
|---|---|---|---|---|---|---|
| Apple | known | 250 | 250 (100%) | 0 | 0 | 161 |
| Peach | known | 250 | 250 (100%) | 0 | 0 | 56 |
| Tomato | known | 250 | 150 (60%) | 100 (`GH_HL Leaf N.k` files; not in any sheet) | 0 | 122 (matched only) |
| **Corn** | known | 250 | **0 (0%)** | 250 (no sheet upstream) | 0 | n/a |
| **Grape** | known | 250 | **0 (0%)** | 250 (no healthy sheet upstream) | 0 | n/a |
| Potato | unknown | 150 | 150 (100%) | 0 | 0 | 38 |
| Bell pepper | unknown | 150 | 150 (100%) | 0 | 0 | 93 |

**Physical leaves crossing the primary split (matched images only):**

| Species | Leaves | Leaves in >1 split | Test images sharing a leaf with train/val |
|---|---|---|---|
| Apple | 161 | 41 (25%) | 24 of 38 |
| Peach | 56 | 42 (75%) | **38 of 38** |
| Tomato (RS_HL) | 122 | 16 (13%) | 9 of 25 |
| Corn, Grape, Tomato GH_HL | n/a | cannot be assessed | cannot be assessed |

So the primary split is **not** leaf-independent. Because the leaf counts are small (a leaf has a median of 4–8 photos), the effect on the known-class numbers is probably largest for Peach. Exact duplicates were removed earlier; hashing could never detect this.

**Is a fully group-aware split feasible for every known class?** No: Corn and Grape have no authoritative IDs and 100 Tomato images are unmatched. No IDs were invented (the number embedded in `GH_HL Leaf 259.1` is not in any sheet and is treated as unverified).

**Does file numbering correspond to acquisition order?** Where it can be checked, yes. In the upstream sheets for Apple, Peach, Bell pepper, Potato and Tomato-RS, `Leaf #` never decreases as the frame number increases (0 violations in every prefix), and for Apple, Peach, Bell pepper and Tomato-RS, 99–100% of leaves occupy a contiguous run of frame numbers (up to 17 photos per leaf). That supports the frame-block heuristic for those classes. It is order within a sequence, not capture timestamps, and it **cannot be verified for Corn, Grape or Tomato-GH**.

**Three splits were evaluated.** All retrained from scratch; primary outputs untouched.

| Split | Basis | Authoritative leaf crossings (matched images) |
|---|---|---|
| primary | stratified, image-level | Apple 41, Peach 42, Tomato 16 leaves cross |
| `variant_blockwise` | frame-block + 5-frame guard (heuristic for all classes) | Apple 1 (3 images), Peach 0, Tomato 0 |
| `variant_leafmap` | **hybrid**: whole leaves where an authoritative ID exists (Apple, Peach, Tomato-RS: 650 images); frame-block heuristic for Corn, Grape and Tomato-GH (600 images, 17 dropped by the guard) | 0, 0, 0 |

The hybrid split is **not fully specimen-independent**: 48% of known images (Corn, Grape, Tomato-GH) are separated only by an unverified heuristic, and capture-session similarity (shared backgrounds, lighting) is not removed by any split here.

| Model | Split | Known acc | Known false rejection | Threshold | Unknown rejected | Potato / Bell pepper | AUROC (leaf-clustered 95% CI) |
|---|---|---|---|---|---|---|---|
| HOG + SVM | primary | 0.989 | 0.016 | 0.564 | 0.130 | 0.107 / 0.153 | 0.775 (0.729–0.820) |
| HOG + SVM | frame-block | 0.957 | 0.064 | 0.614 | 0.193 | 0.147 / 0.240 | 0.748 (0.697–0.799) |
| HOG + SVM | hybrid leaf-map | 0.957 | 0.022 | 0.565 | 0.123 | 0.100 / 0.147 | 0.799 (0.752–0.844) |
| MobileNetV3-S | primary | 1.000 | 0.053 | 0.990 | 0.730 | 0.607 / 0.853 | 0.948 (0.919–0.971) |
| MobileNetV3-S | frame-block | 1.000 | 0.053 | 0.997 | 0.660 | 0.407 / 0.913 | 0.919 (0.884–0.950) |
| MobileNetV3-S | hybrid leaf-map | 0.995 | 0.113 | 0.996 | 0.733 | 0.553 / 0.913 | 0.898 (0.859–0.934) |

**Interpretation (evidence vs hypothesis).**
- *Evidence:* the primary split shares physical leaves between train and test for Apple, Peach and Tomato-RS (Peach: every test image). Primary known-class accuracy and confidence are therefore not leaf-independent.
- *Evidence:* the qualitative conclusions survive in all three splits: MobileNet ranks unknowns far better than HOG+SVM (AUROC 0.90–0.95 vs 0.75–0.80), and about a quarter to a third of unknown leaves are still accepted, with Potato the hardest.
- *Evidence, mixed:* under the hybrid split MobileNet AUROC fell (0.948 → 0.898) and its known false-rejection rate doubled (5.3% → 11.3%, mainly Grape and Peach); HOG+SVM Peach recall fell from 0.974 to 0.861. But MobileNet unknown rejection did not change (0.730 → 0.733) and HOG+SVM AUROC rose (0.775 → 0.799).
- *Hypothesis, not established:* that same-leaf overlap inflated the primary known-class confidence and AUROC. The test, validation and training images, and the validation-derived thresholds all differ between splits, only one seed was run, and several intervals overlap; a lower score by itself does not prove leakage.
- Treat the primary numbers as optimistic by an amount that is not precisely known, and quote the hybrid numbers beside them.

**Statistical independence of the unknown pools.** The 300 "independent" unknown images are 38 Potato leaves (4 photos each) and 93 Bell pepper leaves (median 1, maximum 4 photos). Potato rejection, which is the hardest result, rests on only 38 leaves. A leaf-clustered bootstrap (whole leaves resampled; images without an ID are singletons) widens intervals only moderately (MobileNet unknown rejection: image bootstrap 0.680–0.783 vs leaf-clustered 0.668–0.789 in the primary split), but the clusters for Corn, Grape and Tomato-GH are unknown, so those intervals are still too narrow.

## 3. Open-set experiment audit

Independently recomputed from the saved `.npz` probabilities with separate numpy code (not `src/open_set.py`); every value matched the saved JSON to 1e-9, for both models.

| Check | Result |
|---|---|
| Unknown species never in training | Pass: `role == known` asserted in both training scripts; verified on the manifest (train split only known; unknown split only Potato/Bell pepper); one path/hash per row. |
| Threshold from known validation only | Pass: `np.percentile(val_conf, 5)` recomputed = saved (0.5637 / 0.9897). Computed in `evaluate_model` before any test metric. |
| Test set not used for tuning | Pass: no hyper-parameter or threshold depends on test or unknown data. Early stopping uses validation loss. |
| Accuracy, macro-F1 (hand-computed per-class F1), coverage, false rejection | Pass. |
| Unknown rejection, per species | Pass: pooled 0.130 / 0.730; Potato 0.107 / 0.607, Bell pepper 0.153 / 0.853. |
| AUROC direction | Pass: positive = unknown, score = −confidence; equals the Mann–Whitney probability P(unknown conf < known conf) (ties 0.5); unit tests cover 1.0 / 0.0 / 0.5 cases. |
| Combined-pool risk (accepted unknown = error) | Pass: 0.590 / 0.313 recomputed; hand-verifiable unit test exists. |
| Saved tables/figures consistent | Pass: re-running `src.evaluate` from saved predictions reproduced every JSON/CSV/PNG byte-identically; prediction CSVs match the `.npz` files and threshold. |
| Prior claims | **Verified:** MobileNet accuracy 100% (188/188), unknown rejection 73.0%, AUROC 0.948. |

Corrections and caveats found:
1. **Bootstrap degeneracy.** A percentile bootstrap of an all-correct sample returns [1.0, 1.0]. `src/bootstrap_ci.py` now also reports exact Clopper–Pearson intervals (MobileNet accuracy 188/188: 0.981–1.000). Existing percentile intervals reproduced unchanged. The image bootstrap resamples images as independent, so it ignores leaf clustering (see §2 for a leaf-clustered version), and it holds model, threshold and split fixed (test-sampling noise only). The script also ran on import; it now has `main()`.
2. **Pool risk depends on the mix.** 0.313 reflects a 300:188 unknown:known ratio (61.5% unknown). The notebook shows MobileNet pool risk would be 0.031 at 10% unknown and 0.222 at 50% unknown.
3. **Wording fix:** the old report said accepted unknowns have confidence "near 1.00" for most; only 29 of the 81 are ≥ 0.999 (all are ≥ the 0.990 threshold).
4. **What accepted unknowns are called (MobileNet):** 52 of the 59 accepted Potato images are classified Tomato; accepted Bell pepper (22) are spread over Apple/Peach/Tomato. HOG+SVM calls almost all accepted unknowns Apple (238 of 261).
5. `results_table.md` is plain text, not Markdown (`tabulate` is not installed); content is correct.

## 4. Analysis notebook
`notebooks/leaf_open_set_analysis.ipynb` (14 sections) loads saved outputs only, recomputes metrics with the audited functions and **asserts equality with the saved JSON**. Executed end to end with `jupyter nbconvert --execute` (kernel Python 3.11.15): all cells ran, no errors, all assertions passed, outputs are stored in the file. Needs `requirements-notebook.txt`.

## 5. Prediction interface
| Item | Status |
|---|---|
| `src.predict` CLI, both models, known / rejected-unknown / accepted-unknown (Potato→Tomato) | Tested; thresholds equal saved ones (0.990 / 0.564). |
| CLI with corrupt and missing files | Tested; **a defect was found and fixed** (raw traceback aborting the batch). It now reports per-file errors, continues, exits 1. |
| Upload decoding (`open_upload`: RGB, RGBA→white, grayscale, palette, EXIF orientation, garbage, empty, >50 MP) | Unit-tested (RGB/RGBA/L/P, transparency, garbage, empty). EXIF and the >50 MP limit are implemented but **not unit-tested**. |
| Streamlit upload workflow | Tested with Streamlit `AppTest` against the real `app.py`: a known leaf (accepted, species shown), an unknown leaf (shows "Unknown / Uncertain"), a corrupt file (friendly error, no exception), and the disclaimer. **Not tested in a real browser** (none available). |
| Disclaimer that "Unknown / Uncertain" is confidence-based rejection | Present in the app and CLI. |

Manual browser test (about 5 minutes): `.venv/bin/streamlit run app.py`; (1) upload a file from `data/raw/Apple___healthy/` → species + confidence, green outcome; (2) upload `data/raw/Potato___healthy/…RS_HL 1864.JPG` → expected "Tomato" accepted at ~1.00 (the failure mode), and `…RS_HL 1930.JPG` → "Unknown / Uncertain" (0.964); (3) upload a `.txt` renamed to `.jpg` → red error; (4) upload a phone photo of a non-leaf and of a real leaf, and note the outcome is not expected to be reliable; (5) check the blue note is visible.

## 6. Results regenerated vs audited
- **Audited only (unchanged):** all primary models, predictions, metrics, tables and figures. `src.evaluate` was re-run on saved predictions and reproduced them byte-identically.
- **Retrained and regenerated (new, separate):** `artifacts/variant_blockwise/*` (frame-block heuristic split) and `artifacts/variant_leafmap/*` (hybrid leaf-map split).
- **Text-only edit to a primary artifact:** the `near_duplicate_note` string in `artifacts/metrics/dataset_report.json` claimed PlantVillage has no leaf IDs; the sentence was corrected, no number changed.
- **Retrain reproducibility check:** HOG+SVM and MobileNet retrained on the primary split in a scratch variant: probabilities differed by exactly 0.0 and `open_set_results.json` was identical (same machine, Apple M4 / MPS). CUDA/CPU not tested.
- **Extended:** `bootstrap_ci.json` (added exact intervals only); `leakage_audit.json`/`split_sensitivity.*` now include the hybrid split.

## 7. Reproducibility
- Leaf-mapping sheets: preserved in `data/leaf_maps/` with SHA-256 and git blob SHAs (`PROVENANCE.json`). Dataset provenance: upstream `spMohanty/PlantVillage-Dataset` commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`: all 1,550 local images used are **byte-identical** (SHA-256) to that commit, and the seven folder counts match.
- `uv.lock` was **not** added: the project has no `pyproject.toml`, so `requirements.txt` (pinned) and `environment.txt` (full freeze) remain the lock. `scipy` (used directly) was added to `requirements.txt`.
- Git hygiene: data, checkpoints, venv and caches are ignored; variant checkpoints are ignored too.

## 8. Not tested
Real-browser upload; CUDA/CPU; clean-machine reinstall from scratch (the venv here pre-existed); multiple seeds; any leaf grouping for Corn, Grape and Tomato-GH (no authoritative source exists); macOS-only; EXIF-rotated and very large uploads; fine-tuned or alternative OOD scores; any non-PlantVillage image.
