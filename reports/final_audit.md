# Final audit (2026-10-27)

Scope: scientific validity first, then completeness. Existing work was preserved; primary results were **audited, not regenerated** (see §6). Everything below was run in this repository; nothing is quoted from memory.

## 1. Baseline
- Git: clean tree at `7d8d1ee`; 8 commits. Tests: **13 passed** (Python 3.11.15). Primary artifacts present (`data/metadata.csv`, predictions, checkpoints, metrics, figures).
- Accidental file found and removed: `nothing-0.0.3-py2.py3-none-any.whl` (tracked, a trivial PyPI package, referenced nowhere). Recoverable from history (`git show 7d8d1ee:nothing-0.0.3-py2.py3-none-any.whl`).

## 2. Data leakage investigation

**Question:** can different photographs of the same physical leaf end up in train and test?

**Finding: it cannot be ruled out, and a true specimen-level split is not possible.**

| Evidence | Result |
|---|---|
| Physical-leaf IDs in PlantVillage file names | **None.** Names are `<uuid>___<site prefix> <camera frame number>.JPG`. |
| Explicit leaf numbers | Only the 100 Tomato `GH_HL Leaf N[.k]` files (96 distinct N). Under the primary split 1 leaf number spans two splits (2 images). |
| Frame-number proximity (primary split) | Dense. For Corn, Grape and Peach every val/test image has a train image within 5 frames (median gap 1); Apple and Tomato: 64–74% within 5 frames. This is mostly sampling density, and **adjacency does not prove same leaf**. |
| Visual check (`artifacts/figures/frame_neighbours.png`, seeded random pairs) | Mixed: Apple neighbours are clearly different leaves; some Tomato/Peach/Corn neighbours could be the same leaf in another pose. Not decidable by eye. |
| Exact duplicates | 13 removed (SHA-256), none left. Coarse aHash near-duplicates were inspected earlier and were different leaves. Hashing cannot show specimen independence. |

**Decision.** No group IDs were fabricated. The primary split and primary results are retained unchanged, with an explicit limitation. Because retraining takes seconds, a clearly labelled **sensitivity experiment** was added: a *frame-block split* (`src/grouped_split.py`) that keeps frames within a 50-frame block (and Tomato leaf numbers) together, then drops any image within 5 frames of an image in another split (39 of 1250 known images dropped; minimum cross-split gap 6 frames; verified by test). It uses the same images and the same unknown set. Models were retrained and all metrics/figures regenerated **into `artifacts/variant_blockwise/`** (via `LEAF_VARIANT=blockwise`), never over the primary outputs.

| Model | Split | Known acc | Threshold | Unknown rejected | Potato / Bell pepper rejected | AUROC (95% bootstrap CI) |
|---|---|---|---|---|---|---|
| HOG + SVM | primary | 0.989 | 0.564 | 0.130 | 0.107 / 0.153 | 0.775 (0.730–0.818) |
| HOG + SVM | frame-block | 0.957 | 0.614 | 0.193 | 0.147 / 0.240 | 0.748 (0.702–0.793) |
| MobileNetV3-S | primary | 1.000 | 0.990 | 0.730 | 0.607 / 0.853 | 0.948 (0.928–0.966) |
| MobileNetV3-S | frame-block | 1.000 | 0.997 | 0.660 | 0.407 / 0.913 | 0.919 (0.894–0.942) |

**Interpretation (evidence vs hypothesis).**
- *Evidence:* the qualitative conclusions survive: MobileNet stays far better than HOG+SVM at ranking unknowns (AUROC 0.92 vs 0.75), keeps 100% closed-set accuracy, and still accepts about a third of unknown leaves. Potato remains the hardest unknown.
- *Evidence:* all headline numbers move in the pessimistic direction under the stricter split (HOG known accuracy −3.2 points, MobileNet AUROC −0.03, unknown rejection −7 points, Potato rejection −20 points).
- *Not established:* that the drop is caused by leakage. The test images, validation images, threshold and training set all differ, there is one seed, and the AUROC/accuracy intervals overlap. The Potato drop (0.607 → 0.407) is the one change large enough to look beyond test-sampling noise, but training variance cannot be excluded. The primary numbers are best read as *possibly mildly optimistic*.
- The frame-block split still cannot guarantee specimen independence: the same leaf could be photographed in non-adjacent frames, and similar capture sessions share backgrounds and lighting.

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
1. **Bootstrap degeneracy.** A percentile bootstrap of an all-correct sample returns [1.0, 1.0]. `src/bootstrap_ci.py` now also reports exact Clopper–Pearson intervals (MobileNet accuracy 188/188: 0.981–1.000). Existing percentile intervals reproduced unchanged. The bootstrap resamples images as independent, so it ignores specimen clustering, and it holds model, threshold and split fixed (test-sampling noise only). The script also ran on import; it now has `main()`.
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
- **Retrained and regenerated (new, separate):** `artifacts/variant_blockwise/*` (frame-block split).
- **Retrain reproducibility check:** HOG+SVM and MobileNet retrained on the primary split in a scratch variant: probabilities differed by exactly 0.0 and `open_set_results.json` was identical (same machine, Apple M4 / MPS). CUDA/CPU not tested.
- **Extended:** `bootstrap_ci.json` (added exact intervals only).

## 7. Reproducibility
- Dataset provenance: upstream `spMohanty/PlantVillage-Dataset` commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`: all 1,550 local images used are **byte-identical** (SHA-256) to that commit, and the seven folder counts match.
- `uv.lock` was **not** added: the project has no `pyproject.toml`, so `requirements.txt` (pinned) and `environment.txt` (full freeze) remain the lock. `scipy` (used directly) was added to `requirements.txt`.
- Git hygiene: data, checkpoints, venv and caches are ignored; variant checkpoints are ignored too.

## 8. Not tested
Real-browser upload; CUDA/CPU; clean-machine reinstall from scratch (the venv here pre-existed); multiple seeds; macOS-only; EXIF-rotated and very large uploads; fine-tuned or alternative OOD scores; any non-PlantVillage image.
