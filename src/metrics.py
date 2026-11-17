
"""Patient-level discrimination and calibration metrics."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)


def compute_metrics(y_true, y_score, is_probability=True):
    """Discrimination always; calibration only for probability outputs.

    Raw clinical scores such as SOFA are not probabilities, so Brier and
    calibration terms are undefined for them and returned as NaN.
    """
    out = {
        'auroc': float(roc_auc_score(y_true, y_score)),
        'auprc': float(average_precision_score(y_true, y_score)),
    }
    if not is_probability:
        out.update(brier=np.nan, cal_slope=np.nan, cal_intercept=np.nan)
        return out

    out['brier'] = float(brier_score_loss(y_true, y_score))
    eps = 1e-7
    p = np.clip(y_score, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        lr.fit(logit, y_true)
        out['cal_slope'] = float(lr.coef_[0][0])
        out['cal_intercept'] = float(lr.intercept_[0])
    except Exception:
        out['cal_slope'] = out['cal_intercept'] = np.nan
    return out
