# Can Your AI Admit It Doesn't Know? — Open-Set Leaf Classification

A controlled open-set recognition experiment on PlantVillage leaves. Two models are trained on **5 known species**
(Apple, Corn, Grape, Peach, Tomato) and evaluated on those plus **2 species never seen in training** (Potato, Bell pepper).
A prediction is *rejected* ("Unknown / Uncertain") when its maximum class probability falls below a threshold chosen
**on known validation images only**.

- Model A: HOG features + RBF-SVM (classical baseline)
- Model B: MobileNetV3-Small, ImageNet-pretrained, frozen backbone, retrained 5-way head
- **Read first:** [reports/final_audit.md](reports/final_audit.md) (leakage investigation, metric audit, what was and was not tested) · notebook: [notebooks/leaf_open_set_analysis.ipynb](notebooks/leaf_open_set_analysis.ipynb)
- Full write-up: [reports/experiment_report.md](reports/experiment_report.md) · limitations: [reports/limitations.md](reports/limitations.md) · slides/viva: [reports/presentation_outline.md](reports/presentation_outline.md) · plan: [PROJECT_PLAN.md](PROJECT_PLAN.md)

## Headline results (single seed 42; known test n=188, unknown test n=300)

| | HOG + SVM | MobileNetV3-Small |
|---|---|---|
| Rejection threshold (5th pct of val confidence) | 0.564 | 0.990 |
| Known test accuracy / macro-F1 | 0.989 / 0.990 | 1.000 / 1.000 |
| Known coverage (accepted) / false rejection | 0.984 / 0.016 | 0.947 / 0.053 |
| Accuracy on accepted known | 0.989 | 1.000 |
| **Unknown species rejected** / wrongly accepted | **0.130** / 0.870 | **0.730** / 0.270 |
| AUROC known-vs-unknown (unknown = positive) | 0.775 (95% CI 0.73–0.82) | 0.948 (0.93–0.97) |
| Pool coverage / pool risk at threshold | 0.914 / 0.590 | 0.531 / 0.313 |

Both models classify the five known species almost perfectly; the difference is in *knowing what they don't know*.
Even the better model wrongly accepts about a quarter of unknown leaves (Potato is harder to reject than Bell pepper).
Figures: `artifacts/figures/`. Metrics: `artifacts/metrics/`. These numbers were independently recomputed from the saved predictions during the final audit. MobileNet accuracy has an exact 95% interval of 0.981–1.000 (the percentile bootstrap degenerates to [1, 1]).

**Caveat on leakage.** PlantVillage has no physical-leaf IDs, so the image-level split may place photographs of the same leaf in train and test. A retrained frame-block sensitivity split (`artifacts/variant_blockwise/`) gives somewhat lower numbers (MobileNet unknown rejection 0.66, AUROC 0.919; HOG+SVM known accuracy 0.957) with the same conclusions; see `reports/final_audit.md`. Results are from controlled lab-style images and do not establish field performance.

## Setup

```bash
uv venv --python 3.11 .venv            # or any Python 3.11 venv
uv pip install --python .venv/bin/python -r requirements.txt   # exact versions tested are in environment.txt
uv pip install --python .venv/bin/python -r requirements-notebook.txt   # only to run the notebook
```
There is no `pyproject.toml`, hence no `uv.lock`; `requirements.txt` (pinned) and `environment.txt` (full freeze) are the lock.
Tested on macOS arm64 (Apple M4, PyTorch MPS). CUDA/CPU are auto-detected; no code changes needed.

## Get the data

PlantVillage `raw/color` (unaugmented) from https://github.com/spMohanty/PlantVillage-Dataset, upstream commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` (the 1,550 images used were verified byte-identical to this commit; same source as the
[TFDS entry](https://www.tensorflow.org/datasets/catalog/plant_village)). Only seven folders are needed; they must end up in `data/raw/`:

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git /tmp/pv   # any scratch dir
cd /tmp/pv && git sparse-checkout set --no-cone "raw/color/Apple___healthy" "raw/color/Corn_(maize)___healthy" \
  "raw/color/Grape___healthy" "raw/color/Peach___healthy" "raw/color/Tomato___healthy" \
  "raw/color/Potato___healthy" "raw/color/Pepper,_bell___healthy" && git checkout
mkdir -p <this repo>/data/raw && cp -R /tmp/pv/raw/color/* <this repo>/data/raw/
```
(The first attempt may fail with a transient network error; retrying worked.) Raw data and checkpoints are git-ignored. The sparse clone above fetches the current upstream `master`; to pin the audited commit run `git fetch --depth 1 origin 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a && git checkout FETCH_HEAD` before copying (the fetch-by-SHA step itself was not tested). Check your copy with:

```bash
.venv/bin/python -m src.verify_data    # 1550/1550 files match the manifest
```

## Run the full pipeline (≈1–2 minutes on an M4)

```bash
PY=.venv/bin/python
$PY -m src.data               # metadata + split manifest -> data/metadata.csv, artifacts/metrics/data_build_log.json
$PY -m src.eda                # dataset report + montage -> artifacts/metrics/dataset_report.json
$PY -m src.train_baseline     # HOG+SVM  -> artifacts/predictions/hog_svm.npz
$PY -m src.train_mobilenet    # MobileNetV3-Small (downloads ImageNet weights) -> artifacts/predictions/mobilenet.npz
$PY -m src.evaluate           # threshold from val, open-set metrics, all figures
$PY -m src.stress_test        # optional synthetic brightness/blur test
$PY -m src.bootstrap_ci       # optional 95% bootstrap + exact binomial CIs
$PY -m pytest -q              # 21 tests
```

Leakage sensitivity experiment (separate outputs; the primary results are not touched):
```bash
$PY -m src.grouped_split                                  # data/metadata_blockwise.csv (frame-block split + guard band)
export LEAF_VARIANT=blockwise                             # redirects every script below to artifacts/variant_blockwise/
$PY -m src.train_baseline && $PY -m src.train_mobilenet && $PY -m src.evaluate && $PY -m src.bootstrap_ci
unset LEAF_VARIANT
$PY -m src.leakage_audit                                  # leakage_audit.json, split_sensitivity.md, frame_neighbours.png
```

Notebook (loads saved outputs, no retraining; assertions compare recomputed and saved metrics):
```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/leaf_open_set_analysis.ipynb
```
There is no separate "quick run": the whole experiment is small enough that the quick and full runs are identical. A full re-run on the same machine (Apple M4, MPS) reproduced all predictions and metrics identically (re-checked in the final audit: probability arrays differ by exactly 0.0); only wall-clock timings differ. Other hardware (CUDA/CPU) and a from-scratch environment rebuild were not tested and may differ slightly.

## Inference

```bash
.venv/bin/python -m src.predict path/to/leaf.jpg [--model mobilenet|hog_svm]
.venv/bin/streamlit run app.py
```
Both use the same preprocessing and the frozen threshold stored in `artifacts/metrics/open_set_results.json`.
Rejection is a confidence-based safeguard, not proof that a plant is botanically unknown.

## Data design and protocol (summary)

- Folder → label map: Apple=`Apple___healthy`, Corn=`Corn_(maize)___healthy`, Grape=`Grape___healthy`, Peach=`Peach___healthy`,
  Tomato=`Tomato___healthy`; unknown: Potato=`Potato___healthy`, Bell pepper=`Pepper,_bell___healthy`. Labels: Apple 0, Corn 1, Grape 2, Peach 3, Tomato 4.
- Healthy leaves only (target = species, not disease). 13 exact duplicates (SHA-256) removed from Apple/Tomato before sampling; 0 corrupt files.
- 250 images/known species (seeded sample) split 70/15/15 stratified at **image level** (no leaf IDs exist; see caveat above) → 875 train / 187 val / 188 test; 150 images each of Potato and Bell pepper, evaluation only.
- All images are 256×256 RGB. Unknown species never enter training (asserted in code, tested).
- Threshold = 5th percentile of max-probability on known **validation** images, frozen before test metrics are computed.

## Repository layout

```
configs/experiment.json   all settings and seed        src/    data, preprocessing, eda, open_set, train_*, evaluate, predict, stress_test,
data/metadata.csv         split manifest (committed)          bootstrap_ci, grouped_split, leakage_audit, verify_data, paths
data/metadata_blockwise.csv  sensitivity split               tests/  pytest suite (21 tests)
artifacts/{metrics,figures,logs,predictions}                  reports/ final_audit, experiment report, limitations, presentation outline
artifacts/variant_blockwise/  sensitivity results              notebooks/ analysis notebook
app.py                    Streamlit demo
```

## What was tested
Tested: unit tests (21), independent metric recomputation, notebook execution, CLI on known/unknown/corrupt inputs, the Streamlit upload workflow via Streamlit's `AppTest`. **Not tested:** a real browser session (manual steps are in `reports/final_audit.md` §5), CUDA/CPU, multiple seeds, field photographs.
