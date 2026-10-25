import numpy as np

from src.open_set import (UNKNOWN_LABEL, accept_mask, auroc_unknown_vs_known, combined_risk_at_threshold, decide,
                          known_metrics, risk_coverage, select_threshold, unknown_metrics)

CL = ["a", "b", "c"]


def test_threshold_is_validation_percentile():
    conf = np.linspace(0.0, 1.0, 101)
    assert np.isclose(select_threshold(conf, 5), 0.05)


def test_boundary_is_accepted_and_below_rejected():
    probs = np.array([[0.7, 0.2, 0.1], [0.69, 0.2, 0.11]])
    out = decide(probs, 0.7, CL)
    assert out[0][2] == "a" and out[1][2] == UNKNOWN_LABEL
    assert accept_mask([0.5], 0.5)[0] and not accept_mask([0.4999], 0.5)[0]


def test_decide_output_format():
    (pred, conf, outcome), = decide(np.array([[0.1, 0.8, 0.1]]), 0.5, CL)
    assert pred == "b" and np.isclose(conf, 0.8) and outcome == "b"


def test_known_and_unknown_metrics_hand_computed():
    probs = np.array([[.9, .05, .05], [.4, .5, .1], [.1, .1, .8], [.34, .33, .33]])
    y = np.array([0, 0, 2, 1])  # preds: 0,1,2,0 -> 2 of 4 correct
    m = known_metrics(probs, y, threshold=0.5)  # accepted: rows 0,1,2 (conf .9,.5,.8)
    assert m["accuracy"] == 0.5 and m["coverage"] == 0.75 and np.isclose(m["false_rejection_rate"], 0.25)
    assert np.isclose(m["accuracy_on_accepted"], 2 / 3)
    u = unknown_metrics(np.array([[.9, .05, .05], [.4, .3, .3]]), 0.5)
    assert u["rejection_rate"] == 0.5 and u["false_acceptance_rate"] == 0.5


def test_auroc_direction_unknown_positive():
    assert auroc_unknown_vs_known(np.array([.9, .95, .99]), np.array([.3, .4, .5])) == 1.0
    assert auroc_unknown_vs_known(np.array([.3, .4]), np.array([.9, .95])) == 0.0
    assert auroc_unknown_vs_known(np.array([.5, .5]), np.array([.5, .5])) == 0.5


def test_risk_coverage_counts_accepted_unknown_as_error():
    kp = np.array([[.9, .1], [.8, .2]]); y = np.array([0, 0])
    up = np.array([[.95, .05], [.55, .45]])
    r = combined_risk_at_threshold(kp, y, up, 0.8)  # accepts .9,.8,.95 ; unknown .95 is an error
    assert np.isclose(r["coverage"], 0.75) and np.isclose(r["risk"], 1 / 3)
    conf = np.array([.9, .8, .95, .55]); correct = np.array([1, 1, 0, 0])
    th, cov, risk = risk_coverage(conf, correct, thresholds=[0.0, 0.99])
    assert cov[0] == 1.0 and np.isclose(risk[0], 0.5) and cov[1] == 0.0 and np.isnan(risk[1])
