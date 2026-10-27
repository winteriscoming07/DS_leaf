# Project plan — Open-set leaf classification with confidence-based rejection

> Original plan, written before the experiments. It is kept for reference; what was actually done, including the later leakage audit and the two split-sensitivity experiments, is in [README.md](README.md) and `reports/`.

## Research question
When a leaf image belongs to a plant species never seen in training, can the model recognise that it may not know the answer?
- RQ1: classical (HOG + SVM) vs pretrained deep model (MobileNetV3-Small) on 5 known species.
- RQ2: can maximum-probability confidence separate known from withheld species?
- RQ3: risk–coverage trade-off of accepting more vs fewer predictions.

## Dataset design
- Source: PlantVillage `raw/color` (unaugmented), GitHub mirror spMohanty/PlantVillage-Dataset (same source as the TFDS `plant_village` entry). Only the 7 *healthy* folders are used; the target is species, not disease.
- Known (trained on): Apple, Corn (maize), Grape, Peach, Tomato. Cap 250 images/species.
- Unknown (withheld, evaluation only): Potato, Bell pepper. Cap 150 images/species.
- Seed 42. Stratified 70/15/15 split of known species; exact duplicates (SHA-256) removed before splitting. Unknown images get role `unknown`, split `unknown_test`.

## Protocol
1. Inspect data (counts, sizes, corrupt files, duplicates, near-duplicates via average hash, montages).
2. Save split manifest (`data/metadata.csv`) before fitting anything; both models share it.
3. Model A: HOG + RBF-SVM (probability=True). Model B: MobileNetV3-Small, ImageNet weights, frozen features, new 5-way head; best checkpoint by validation loss.
4. Confidence = max class probability. Threshold = 5th percentile of known-validation confidence, frozen before test evaluation.
5. Report known accuracy/macro-F1, coverage, false rejection, accepted accuracy, unknown rejection / false acceptance, AUROC (positive = unknown, score = −max-confidence), risk–coverage curves.
6. Failure galleries; optional synthetic brightness/blur stress test.

## Deliverables
README, config, src/, tests/, figures + metrics in `artifacts/`, reports (experiment report, limitations, presentation outline + viva notes), CLI inference; Streamlit app (completed).

## Known limitations (anticipated)
PlantVillage is lab-style, uniform-background imagery; small samples; softmax confidence is not calibrated; two unknown species only; single seed; results do not imply field generalisation.
