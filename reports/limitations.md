# Limitations and assumptions

**What this experiment is:** a controlled lab-style comparison on PlantVillage. **What it is not:** evidence of field performance.

1. **Dataset realism.** PlantVillage leaves are single leaves photographed on uniform grey/pink backgrounds. Near-perfect closed-set accuracy (98.9–100%) reflects an easy task, not a solved problem. Results say nothing about cluttered field photos, multiple leaves, other lighting or cameras.
2. **Possible same-leaf leakage.** PlantVillage has no leaf IDs; several photos of one physical leaf may fall into different splits, inflating accuracy. Exact duplicates were removed (13); a coarse aHash check flagged candidates that visual inspection showed to be different leaves, but this cannot rule out same-leaf pairs.
3. **Only two unknown species, both from the same family as Tomato.** Potato, Bell pepper and Tomato are all Solanaceae. This may make the open-set task hard in a way that does not generalise to unrelated plants. (Hypothesis, not tested.) The unknown set is also not "unknown" to the pretrained backbone: ImageNet pretraining may have seen related plants.
4. **Softmax is not calibrated.** Maximum probability is a heuristic confidence. MobileNet probabilities saturate near 1.0 (threshold 0.990); HOG+SVM uses Platt-scaled probabilities from a different mechanism. Threshold values and scores are **not directly comparable** across models; only the protocol and the ranking-based AUROC are.
5. **Threshold rule.** The 5th-percentile rule fixes validation acceptance at ~95% by construction; it is not optimised for unknown rejection. A different rule (or a method such as energy/ODIN/Mahalanobis) could do better. The rule was fixed in advance and not tuned on test data.
6. **Small samples, one seed.** 188 known and 300 unknown test images; one split, one training run. Bootstrap CIs capture test-sampling noise only, not split/seed/training variance. The HOG vs MobileNet closed-set difference (2 errors) is not meaningful.
7. **Pool risk depends on class mix.** Combined risk–coverage uses 188 known vs 300 unknown images; with more unknowns, risk is higher. Absolute pool-risk values are specific to this ratio. The known-only curve is shown separately.
8. **Classical baseline is minimal.** HOG on grayscale with fixed SVM hyper-parameters, no tuning, no colour features. It is a transparent baseline, not the best classical model.
9. **Model scope.** Frozen backbone only; no fine-tuning or architecture comparison was run. Augmentation is mild and was not ablated.
10. **Synthetic stress test.** Brightness and blur applied to copies of known test images only; it is not evidence about real degradations, and unknown images were not perturbed.
11. **Explanations.** No experiment here shows *which* visual features drive decisions. Statements such as "similar leaf shape" are observations of gallery images, not causal claims.
12. **Reproducibility.** Tested on one machine (Apple M4, MPS); the SVM uses a deprecated `probability=True` path in scikit-learn 1.9 (works; pinned).
13. **Untested:** CUDA/CPU runs, clean-machine reinstall from scratch, the Streamlit upload interaction (only that the app loads without exceptions was checked).
