# Experiment report — Can the model admit it doesn't know?

All numbers are from `artifacts/metrics/*.json` produced by the code in this repository (seed 42, single run).

## 1. Design
- Data: PlantVillage `raw/color`, healthy leaves. Known = Apple, Corn, Grape, Peach, Tomato (250 each, after removing 13 exact duplicates and sampling with seed 42). Unknown = Potato, Bell pepper (150 each, evaluation only). 256×256 RGB, no corrupt files.
- Known split 70/15/15 stratified: 875 train / 187 val / 188 test; shared by both models (manifest `data/metadata.csv`, written before any model was fitted).
- Model A: HOG (9 orientations, 16 px cells, 2×2 blocks, 128×128 grayscale → 1764 features) + RBF-SVM (C=10, gamma=scale, Platt probabilities), no tuning.
- Model B: MobileNetV3-Small, ImageNet weights (verified downloaded and used), backbone frozen, classifier head retrained, Adam lr 1e-3, batch 32, mild augmentation (train only), best checkpoint by validation loss (epoch 8; early stopping at epoch 12). Trained on Apple MPS in 21 s.
- Rejection: confidence = max class probability; threshold = 5th percentile of known-validation confidence (HOG+SVM 0.564; MobileNet 0.990); accept if confidence ≥ threshold. Frozen before test metrics were computed (`src/evaluate.py`).

## 2. RQ1 — classical vs deep (known test, n=188)
| | HOG + SVM | MobileNetV3-Small |
|---|---|---|
| Accuracy | 0.989 (2 errors: a Grape and a Peach predicted as Apple) | 1.000 |
| Macro-F1 | 0.990 | 1.000 |
| Inference per image | ~2.0 ms incl. HOG (CPU) | ~1.7 ms excl. decode (MPS) |
| Fit/train time | 2.2 s | 21.3 s |

Both are near ceiling; with 188 test images the 2-error gap is not statistically meaningful. PlantVillage species are easy to separate (plain backgrounds, distinct shapes/colours). Honest answer to RQ1: on this data a transparent HOG+SVM is nearly as accurate as transfer learning; the difference shows up in RQ2.

## 3. RQ2 — rejecting unseen species
| | HOG + SVM | MobileNet |
|---|---|---|
| Unknown rejected (Potato / Bell pepper) | 0.130 (0.107 / 0.153) | 0.730 (0.607 / 0.853) |
| Unknown wrongly accepted | 0.870 | 0.270 |
| AUROC unknown-vs-known (95% bootstrap CI) | 0.775 (0.73–0.82) | 0.948 (0.93–0.97) |
| Known false rejection | 0.016 (3 of 188) | 0.053 (10 of 188) |
| Known coverage at test (val target was ≈0.947 for both) | 0.984 | 0.947 |

HOG+SVM's probabilities are almost as confident for unknown leaves as for known ones, so rejection barely helps (13%). MobileNet's confidence separates the groups much better (AUROC 0.95), yet at the validation-chosen threshold it still accepts 27% of unknown leaves, most of them with confidence near 1.00 (gallery: Potato leaves labelled Tomato at 1.00). The "95% familiar acceptance" target applies to validation; realised known-test coverage was 98.4% and 94.7%. AUROC is threshold-free and shows MobileNet ranks unknowns below knowns far better than HOG+SVM; the operating-point rejection rate depends on the threshold rule.

Note on comparability: the two models' confidences come from different mechanisms (Platt-scaled SVM vs. softmax), so only the protocol and AUROC are comparable, not raw probability values.

## 4. RQ3 — risk vs coverage (`artifacts/figures/risk_coverage.png`)
- Pool (188 known + 300 unknown; accepted unknown counts as an error). At the chosen thresholds: HOG+SVM coverage 0.914, risk 0.590; MobileNet coverage 0.531, risk 0.313. With no rejection, risk is 300/488 = 0.615 for both (every accepted unknown is an error), so rejection reduces risk from 0.615 to 0.313 for MobileNet but only to 0.590 for HOG+SVM. MobileNet's risk falls to roughly 3–5% around 20–30% coverage — only a minority of images are accepted at that level.
- Known only: MobileNet has zero errors at every threshold (risk flat at 0); rejecting known images there costs coverage without improving accuracy. HOG+SVM's known-only risk stays around 0.5–1% (its 2 errors).
- Plain-language trade-off: a stricter threshold rejects more unfamiliar leaves but also refuses correct predictions on familiar ones (all 10 known-rejected MobileNet images were classified correctly). On this data rejecting is a useful safeguard only because unknown images are plentiful and known accuracy is already saturated.

## 5. Error analysis (`failure_gallery_*.png`, seeded random examples)
- MobileNet: 178 known correct+accepted, 0 known wrong, 10 known rejected (confidence 0.87–0.95, all correctly predicted), 81 unknown accepted, 219 unknown rejected. Accepted unknowns shown are Potato leaves predicted Tomato at ≈1.00 — Potato and Tomato are both Solanaceae and the shown leaves look visually similar to tomato leaves, a plausible but **untested** explanation.
- HOG+SVM: 183 known correct+accepted, 2 known wrong, 3 known rejected, 261 unknown accepted, 39 unknown rejected; in the sampled gallery the accepted unknowns were mostly assigned to Apple.
- No experiment here identifies which image features the networks use.

## 6. Synthetic stress test (known test images only; `stress_test.png`; thresholds frozen)
| Perturbation | HOG+SVM acc / coverage | MobileNet acc / coverage |
|---|---|---|
| none | 0.989 / 0.984 | 1.000 / 0.947 |
| brightness ×0.4 | 0.979 / 0.989 | 0.995 / 0.920 |
| brightness ×1.6 | 0.926 / 0.952 | 1.000 / 0.926 |
| blur radius 2 px | 0.686 / 0.777 | 0.984 / 0.803 |
| blur radius 4 px | 0.207 / 0.846 | 0.926 / 0.505 |

At heavy blur HOG+SVM is at chance level (0.207) yet still accepts 85% of images (confidently wrong), whereas MobileNet's coverage falls to 51% — it abstains more as inputs degrade. This is a synthetic test, not field evidence.

## 7. Conclusion
On PlantVillage, a pretrained MobileNetV3-Small with max-probability rejection separates unseen from known species far better than HOG+SVM (AUROC 0.95 vs 0.78), but a fixed 95%-validation-acceptance threshold still lets 27% of unseen-species images through — the model cannot reliably "admit it doesn't know", especially for Potato, a close relative of Tomato. See `limitations.md` for what this does not establish.

## 8. Checks performed
Unknown species excluded from training (asserted in both training scripts; split tests); threshold computed from validation confidences only; 13 pytest tests pass (data discovery, corrupt/duplicate handling, reproducible/isolated splits, preprocessing, threshold, boundary acceptance, metrics incl. AUROC direction, risk–coverage, inference format). Not done: multiple seeds, fine-tuning, alternative OOD scores, CUDA/CPU runs, interactive upload testing of the Streamlit app.
