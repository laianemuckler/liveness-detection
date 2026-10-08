"""
Evaluation and experiment logging functions.
"""

import os
import csv
from datetime import date

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from src.config import EXPERIMENT_LOG_PATH, LABEL_LIVE, LABEL_SPOOF


def compute_metrics(y, y_scores, y_pred):
    """
    Computes the standard FAS metrics from ground truth, P(spoof) scores
    and hard predictions. Shared by every model type (SVM, CNN, ViT), so
    all methods are scored with exactly the same definitions.
    Returns a dict with y_pred, y_scores, FAR, FRR, HTER, AUC, EER.

    FAR (False Acceptance Rate): spoof wrongly classified as live.
    FRR (False Rejection Rate):  live wrongly classified as spoof.
    HTER: average of FAR and FRR, at the model's default decision threshold
    (the one that produced y_pred).
    EER (Equal Error Rate): point where FAR and FRR are equal, scanning
    all possible thresholds (independent of the model's default cutoff).
    """
    y = np.array(y)
    y_pred = np.array(y_pred)

    far = np.sum((y_pred == LABEL_LIVE) & (y == LABEL_SPOOF)) / np.sum(y == LABEL_SPOOF)
    frr = np.sum((y_pred == LABEL_SPOOF) & (y == LABEL_LIVE)) / np.sum(y == LABEL_LIVE)
    hter = (far + frr) / 2
    auc = roc_auc_score(y, y_scores)

    fpr, tpr, thresholds = roc_curve(y, y_scores, pos_label=LABEL_SPOOF)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float(fpr[eer_idx])
    eer_threshold = float(thresholds[eer_idx])

    return {
        'y_pred': y_pred,
        'y_scores': y_scores,
        'FAR': far,
        'FRR': frr,
        'HTER': hter,
        'AUC': auc,
        'EER': eer,
        'EER_threshold': eer_threshold,
    }


def evaluate(model, scaler, X, y):
    """
    Runs a scikit-learn model on (X, y) and computes standard FAS metrics
    (see compute_metrics for the definitions).
    Returns a dict with y_pred, y_scores, FAR, FRR, HTER, AUC, EER.
    """
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_scores = model.predict_proba(X_scaled)[:, 1]  # P(spoof)

    return compute_metrics(y, y_scores, y_pred)


def log_experiment(exp_id, metodo, feature_config, modelo_config,
                    hter_val=None, auc_val=None, hter_test=None, auc_test=None,
                    eer_val=None, eer_test=None, obs=''):
    """
    Appends one row to the central experiment_log.csv (see src/config.py
    for its path). Creates the file with a header if it doesn't exist yet.

    Leave hter_test/auc_test/eer_test as None while still comparing configs
    on the validation set; fill them in only for the final chosen config,
    evaluated once on the test set.
    """
    file_exists = os.path.isfile(EXPERIMENT_LOG_PATH)

    row = {
        'exp_id': exp_id,
        'metodo': metodo,
        'feature_config': feature_config,
        'modelo_config': modelo_config,
        'HTER_val': f"{hter_val:.4f}" if hter_val is not None else '',
        'AUC_val': f"{auc_val:.4f}" if auc_val is not None else '',
        'EER_val': f"{eer_val:.4f}" if eer_val is not None else '',
        'HTER_test': f"{hter_test:.4f}" if hter_test is not None else '',
        'AUC_test': f"{auc_test:.4f}" if auc_test is not None else '',
        'EER_test': f"{eer_test:.4f}" if eer_test is not None else '',
        'data': date.today().isoformat(),
        'obs': obs,
    }

    os.makedirs(os.path.dirname(EXPERIMENT_LOG_PATH), exist_ok=True)

    with open(EXPERIMENT_LOG_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged experiment '{exp_id}' to {EXPERIMENT_LOG_PATH}")