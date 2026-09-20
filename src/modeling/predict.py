"""
Evaluation and experiment logging functions.
"""

import os
import csv
from datetime import date

import numpy as np
from sklearn.metrics import roc_auc_score

from src.config import EXPERIMENT_LOG_PATH, LABEL_LIVE, LABEL_SPOOF


def evaluate(model, scaler, X, y):
    """
    Runs the model on (X, y) and computes standard FAS metrics.
    Returns a dict with y_pred, y_scores, FAR, FRR, HTER, AUC.

    FAR (False Acceptance Rate): spoof wrongly classified as live.
    FRR (False Rejection Rate):  live wrongly classified as spoof.
    HTER: average of FAR and FRR.
    """
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_scores = model.predict_proba(X_scaled)[:, 1]  # P(spoof)

    y = np.array(y)
    y_pred = np.array(y_pred)

    far = np.sum((y_pred == LABEL_SPOOF) & (y == LABEL_LIVE)) / np.sum(y == LABEL_LIVE)
    frr = np.sum((y_pred == LABEL_LIVE) & (y == LABEL_SPOOF)) / np.sum(y == LABEL_SPOOF)
    hter = (far + frr) / 2
    auc = roc_auc_score(y, y_scores)

    return {
        'y_pred': y_pred,
        'y_scores': y_scores,
        'FAR': far,
        'FRR': frr,
        'HTER': hter,
        'AUC': auc,
    }


def log_experiment(exp_id, metodo, feature_config, modelo_config,
                    hter_val=None, auc_val=None, hter_test=None, auc_test=None,
                    obs=''):
    """
    Appends one row to the central experiment_log.csv (see src/config.py
    for its path). Creates the file with a header if it doesn't exist yet.

    Leave hter_test/auc_test as None while still comparing configs on
    the validation set; fill them in only for the final chosen config,
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
        'HTER_test': f"{hter_test:.4f}" if hter_test is not None else '',
        'AUC_test': f"{auc_test:.4f}" if auc_test is not None else '',
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