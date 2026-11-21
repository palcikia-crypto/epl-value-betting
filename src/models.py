"""Модели: XGBoost (multi:softprob) + Logistic Regression baseline.

Подбор гиперпараметров — на train+validation с TimeSeriesSplit (НЕ обычный KFold).
Калибровка — isotonic на validation set (cv='prefit').
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import sklearn
from packaging.version import Version
if Version(sklearn.__version__) >= Version("1.6"):
    from sklearn.frozen import FrozenEstimator
else:
    FrozenEstimator = None


def _frozen(est):
    return FrozenEstimator(est) if FrozenEstimator is not None else est


def tune_xgb(X_trainval, y_trainval, n_splits=3):
    base = XGBClassifier(
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, verbosity=0, n_jobs=-1,
    )
    grid = {
        "n_estimators": [200],
        "max_depth": [3, 4],
        "learning_rate": [0.03, 0.08],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [5],
    }
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(base, grid, cv=tscv, scoring="neg_log_loss", n_jobs=-1, refit=True)
    gs.fit(X_trainval, y_trainval)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def fit_logreg(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def calibrate_prefit(model, X_cal, y_cal):
    return CalibratedClassifierCV(_frozen(model), method="isotonic").fit(X_cal, y_cal)


def fit_calibrated_ts(make_model, X, y, n_splits=5):
    """Кросс-валидированная isotonic-калибровка ВНУТРИ X (без раздута валидации):
    cv = TimeSeriesSplit, чтобы калибровочные фолды не заглядывали в прошлое."""
    cv = list(TimeSeriesSplit(n_splits=n_splits).split(X))
    return CalibratedClassifierCV(make_model(), method="isotonic", cv=cv).fit(X, y)


def predict_proba(model, X):
    return model.predict_proba(X)


def pick(preds, threshold, odds):
    """Возвращает список ставок: [(match_key, outcome, odds, prob)]. Выбор — максимум edge."""
    bets = []
    for outcome in (0, 1, 2):
        edge = odds[:, outcome] * preds[:, outcome] - 1.0
        ok = (preds[:, outcome] > (1.0 / odds[:, outcome]) * (1.0 + threshold))
        ok &= np.isfinite(odds[:, outcome]) & (odds[:, outcome] > 1.0)
        for i in np.where(ok)[0]:
            bets.append((i, outcome, odds[i, outcome], preds[i, outcome], edge[i]))
    return bets


def argmax_pick(preds, odds):
    """Ставка на исход с максимальным edge (не более 1 ставки на матч)."""
    best = []
    for i in range(preds.shape[0]):
        e = odds[i] * preds[i] - 1.0
        e[~np.isfinite(odds[i]) | (odds[i] <= 1.0)] = -np.inf
        o = int(np.argmax(e))
        if np.isfinite(e[o]):
            best.append((i, o, float(odds[i, o]), float(preds[i, o]), float(e[o])))
    return best
