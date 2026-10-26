| Model | Split | Known test n | Known acc | Threshold | Unknown rejected | Potato rejected | Bell pepper rejected | AUROC | AUROC 95% CI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HOG + SVM | primary (image-level) | 188 | 0.989 | 0.564 | 0.13 | 0.107 | 0.153 | 0.775 | 0.730-0.818 |
| HOG + SVM | frame-block heuristic + guard | 188 | 0.957 | 0.614 | 0.193 | 0.147 | 0.24 | 0.748 | 0.702-0.793 |
| HOG + SVM | hybrid leaf-map (leaf IDs where available) | 186 | 0.957 | 0.565 | 0.123 | 0.1 | 0.147 | 0.799 | 0.757-0.840 |
| MobileNetV3-Small | primary (image-level) | 188 | 1.0 | 0.99 | 0.73 | 0.607 | 0.853 | 0.948 | 0.928-0.966 |
| MobileNetV3-Small | frame-block heuristic + guard | 188 | 1.0 | 0.997 | 0.66 | 0.407 | 0.913 | 0.919 | 0.894-0.942 |
| MobileNetV3-Small | hybrid leaf-map (leaf IDs where available) | 186 | 0.995 | 0.996 | 0.733 | 0.553 | 0.913 | 0.898 | 0.868-0.926 |
