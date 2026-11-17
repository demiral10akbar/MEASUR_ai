"""Model builders for the AI modules of Table A1 (scikit-learn MLPs, Adam optimizer).

The architectures are indicative starting configurations (manuscript Section 3.3.6)
and should be tuned on the validation set of each facility.
"""
from __future__ import annotations

import copy

import numpy as np
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _mlp_regressor(hidden, seed):
    return MLPRegressor(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                        learning_rate_init=1e-3, alpha=1e-4, random_state=seed)


def make_demand_forecaster(hidden=(32, 16), seed: int = 0) -> Pipeline:
    """Compressed-air or ventilation demand forecaster (MLP regression, MSE loss)."""
    return Pipeline([("scale", StandardScaler()), ("mlp", _mlp_regressor(hidden, seed))])


def make_torque_regressor(hidden=(32, 16), seed: int = 0) -> Pipeline:
    """Mixer torque predictor from motor current, speed and batch phase (MLP regression)."""
    return Pipeline([("scale", StandardScaler()), ("mlp", _mlp_regressor(hidden, seed))])


def make_pump_health_classifier(hidden=(64, 32), seed: int = 0) -> Pipeline:
    """Pump health-state classifier (MLP, softmax output, cross-entropy loss)."""
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                        learning_rate_init=1e-3, alpha=1e-4, random_state=seed)
    return Pipeline([("scale", StandardScaler()), ("mlp", clf)])


def _balanced_resample(X, y, seed):
    """Random oversampling of minority classes (class weighting for MLPClassifier)."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n_max = counts.max()
    idx = np.concatenate([rng.choice(np.flatnonzero(y == c), n_max, replace=True) for c in classes])
    rng.shuffle(idx)
    return X[idx], y[idx]


def fit_mlp_with_early_stopping(pipe: Pipeline, X_train, y_train, X_val, y_val,
                                max_epochs: int = 300, patience: int = 20,
                                balance_classes: bool = False, seed: int = 0):
    """Train the MLP epoch by epoch and stop when the validation loss stops improving.

    Unlike scikit-learn's built-in early_stopping (which draws a random validation
    subset), this uses the explicit chronological or group-wise validation set.
    Returns the fitted pipeline (best epoch) and the validation-loss history.
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)
    scaler: StandardScaler = pipe.named_steps["scale"]
    mlp = pipe.named_steps["mlp"]
    scaler.fit(X_train)
    Xt, Xv = scaler.transform(X_train), scaler.transform(X_val)
    is_clf = isinstance(mlp, MLPClassifier)
    if is_clf and balance_classes:
        Xt, y_train = _balanced_resample(Xt, y_train, seed)
    classes = np.unique(np.concatenate([y_train, y_val])) if is_clf else None
    best, best_loss, wait, history = None, np.inf, 0, []
    for _ in range(max_epochs):
        if is_clf:
            mlp.partial_fit(Xt, y_train, classes=classes)
            loss = log_loss(y_val, mlp.predict_proba(Xv), labels=mlp.classes_)
        else:
            mlp.partial_fit(Xt, y_train)
            loss = mean_squared_error(y_val, mlp.predict(Xv))
        history.append(loss)
        if loss < best_loss - 1e-6:
            best, best_loss, wait = copy.deepcopy(mlp), loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    pipe.steps[-1] = ("mlp", best)
    return pipe, history
