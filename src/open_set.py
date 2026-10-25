"""Confidence-based rejection and open-set metrics (pure numpy/sklearn, no model dependencies)."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

UNKNOWN_LABEL = "Unknown / Uncertain"


def max_confidence(probs):
    return np.asarray(probs).max(axis=1)


def select_threshold(val_conf, percentile=5):
    """Threshold = percentile of confidence on KNOWN validation images only."""
    return float(np.percentile(np.asarray(val_conf), percentile))


def accept_mask(conf, threshold):
    """Accept when confidence >= threshold (boundary value is accepted)."""
    return np.asarray(conf) >= threshold


def decide(probs, threshold, classes):
    """Return (predicted class name, confidence, outcome) per row; outcome is a species or UNKNOWN_LABEL."""
    probs = np.asarray(probs)
    pred, conf = probs.argmax(1), max_confidence(probs)
    acc = accept_mask(conf, threshold)
    return [(classes[p], float(c), classes[p] if a else UNKNOWN_LABEL) for p, c, a in zip(pred, conf, acc)]


def known_metrics(probs, y, threshold):
    """Metrics on known-species images, before and after rejection."""
    probs, y = np.asarray(probs), np.asarray(y)
    pred, conf = probs.argmax(1), max_confidence(probs)
    acc = accept_mask(conf, threshold)
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "coverage": float(acc.mean()), "false_rejection_rate": float(1 - acc.mean()),
           "accuracy_on_accepted": float(accuracy_score(y[acc], pred[acc])) if acc.any() else float("nan")}
    return out


def unknown_metrics(unk_probs, threshold):
    acc = accept_mask(max_confidence(unk_probs), threshold)
    return {"n": int(len(acc)), "rejection_rate": float(1 - acc.mean()), "false_acceptance_rate": float(acc.mean())}


def auroc_unknown_vs_known(known_conf, unknown_conf):
    """Positive class = unknown; anomaly score = -max confidence (higher => more likely unknown)."""
    y = np.r_[np.zeros(len(known_conf)), np.ones(len(unknown_conf))]
    s = -np.r_[known_conf, unknown_conf]
    return float(roc_auc_score(y, s))


def risk_coverage(conf, correct, thresholds=None):
    """Sweep thresholds. coverage = accepted/all evaluated; risk = incorrect/accepted
    (risk is NaN when nothing is accepted). `correct` must be False for unknown images."""
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=bool)
    if thresholds is None:
        thresholds = np.unique(np.r_[conf, conf.max() + 1e-9])
    cov, risk = [], []
    for t in thresholds:
        a = conf >= t
        cov.append(a.mean())
        risk.append((~correct[a]).mean() if a.any() else np.nan)
    return np.asarray(thresholds), np.asarray(cov), np.asarray(risk)


def combined_correctness(known_probs, y_known, unknown_probs):
    """Confidences and correctness for known+unknown pool (unknown is never 'correct' if accepted)."""
    kp, up = np.asarray(known_probs), np.asarray(unknown_probs)
    conf = np.r_[max_confidence(kp), max_confidence(up)]
    correct = np.r_[kp.argmax(1) == np.asarray(y_known), np.zeros(len(up), dtype=bool)]
    return conf, correct


def combined_risk_at_threshold(known_probs, y_known, unknown_probs, threshold):
    conf, correct = combined_correctness(known_probs, y_known, unknown_probs)
    a = accept_mask(conf, threshold)
    return {"coverage": float(a.mean()), "risk": float((~correct[a]).mean()) if a.any() else float("nan"),
            "n_known": len(known_probs), "n_unknown": len(unknown_probs)}
