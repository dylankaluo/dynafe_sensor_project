from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data import SENSOR_TYPES


def _seconds_to_rows(seconds: float, dt: float, min_rows: int = 1) -> int:
    return max(min_rows, int(round(seconds / max(dt, 1e-9))))


def _sensor_type_index() -> Dict[str, List[int]]:
    type_to_indices: Dict[str, List[int]] = {}
    for idx, typ in enumerate(SENSOR_TYPES):
        type_to_indices.setdefault(typ, []).append(idx)
    return type_to_indices


def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clean without using future information."""
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)
    X = X.loc[:, ~X.columns.duplicated()]
    return X.astype(np.float32)


def _rename_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Return a copy with new columns, avoiding pandas column-alignment side effects."""
    return pd.DataFrame(df.values, columns=columns, index=df.index)


def _frame_from_values(values: np.ndarray, columns: List[str], index) -> pd.DataFrame:
    return pd.DataFrame(values, columns=columns, index=index)


def build_dynafe_features(
    df: pd.DataFrame,
    dt: float,
    mode: str = "full",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Causal dynamic feature engineering for DynaFE-Net.

    The features are designed for dynamic chemical sensor data:

    1. log-resistance transform:
       reduce sensor scale dominance and preserve positive-response structure.

    2. local dynamic descriptors:
       first difference, second difference, rising component, recovery component.

    3. causal multi-scale memory:
       lag, lag difference, slope, EMA, rolling mean, rolling std.

    4. sensor-type structural descriptors:
       because the 16-channel array contains 4 sensor types with repeated units.

    5. cross-sensitivity descriptors:
       type contrasts and type interactions for mixed-gas response coupling.

    mode:
        compact: fewer windows, faster
        full:    richer features, recommended for final experiments
    """
    sensor_cols = [f"s{i:02d}" for i in range(1, 17)]
    sensors = df[sensor_cols].astype(float).values

    eps = 1e-6

    log_res_values = np.log1p(40000.0 / np.maximum(sensors, eps))
    log_res = pd.DataFrame(
        log_res_values,
        columns=[f"logR_{c}" for c in sensor_cols],
        index=df.index,
    ).astype(np.float32)

    base_cols = list(log_res.columns)

    if mode == "compact":
        lag_seconds = [1, 5, 15, 30, 60]
        ema_seconds = [5, 15, 60]
        rolling_seconds = [5, 30, 60]
        type_slope_seconds = [30, 60]
    elif mode == "full":
        lag_seconds = [1, 2, 5, 10, 30, 60, 90, 120]
        ema_seconds = [2, 5, 15, 30, 60, 120]
        rolling_seconds = [5, 15, 30, 60, 90, 120]
        type_slope_seconds = [30, 60, 120]
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    features: List[pd.DataFrame] = []

    # 1) Current state.
    features.append(log_res)

    # 2) Local dynamics.
    d1_raw = log_res.diff().fillna(0.0) / max(dt, eps)
    d1 = _rename_frame(d1_raw, [f"d1ps_{c}" for c in base_cols])

    d2_raw = d1_raw.diff().fillna(0.0) / max(dt, eps)
    d2 = _rename_frame(d2_raw, [f"d2ps_{c}" for c in base_cols])

    d1_pos_raw = d1_raw.clip(lower=0.0)
    d1_pos = _rename_frame(d1_pos_raw, [f"rising_{c}" for c in base_cols])

    d1_neg_raw = -d1_raw.clip(upper=0.0)
    d1_neg = _rename_frame(d1_neg_raw, [f"recovery_{c}" for c in base_cols])

    features.extend([d1, d2, d1_pos, d1_neg])

    # 3) Causal lag, change from lag, and lag-based slope.
    for seconds in lag_seconds:
        rows = _seconds_to_rows(seconds, dt)

        lag_raw = log_res.shift(rows)

        lag = _rename_frame(
            lag_raw,
            [f"lag{seconds}s_{c}" for c in base_cols],
        )

        diff_values = log_res.values - lag_raw.values
        diff = _frame_from_values(
            diff_values,
            [f"diff_lag{seconds}s_{c}" for c in base_cols],
            log_res.index,
        )

        slope_values = diff_values / max(seconds, eps)
        slope = _frame_from_values(
            slope_values,
            [f"slope{seconds}s_{c}" for c in base_cols],
            log_res.index,
        )

        features.extend([lag, diff, slope])

    # 4) EMA memory and deviation from EMA.
    ema_cache: Dict[int, pd.DataFrame] = {}

    for seconds in ema_seconds:
        rows = _seconds_to_rows(seconds, dt, min_rows=2)

        ema_raw = log_res.ewm(span=rows, adjust=False, min_periods=1).mean()
        ema_cache[seconds] = ema_raw

        ema = _rename_frame(
            ema_raw,
            [f"ema{seconds}s_{c}" for c in base_cols],
        )

        dev_values = log_res.values - ema_raw.values
        dev = _frame_from_values(
            dev_values,
            [f"dev_ema{seconds}s_{c}" for c in base_cols],
            log_res.index,
        )

        features.extend([ema, dev])

    if len(ema_seconds) >= 2:
        short_s = ema_seconds[0]
        long_s = ema_seconds[-1]

        contrast_values = ema_cache[short_s].values - ema_cache[long_s].values
        contrast = _frame_from_values(
            contrast_values,
            [f"ema_contrast{short_s}_{long_s}s_{c}" for c in base_cols],
            log_res.index,
        )
        features.append(contrast)

    # 5) Rolling statistics.
    for seconds in rolling_seconds:
        rows = _seconds_to_rows(seconds, dt, min_rows=2)

        mean_raw = log_res.rolling(rows, min_periods=1).mean()
        mean = _rename_frame(
            mean_raw,
            [f"roll_mean{seconds}s_{c}" for c in base_cols],
        )

        std_raw = log_res.rolling(rows, min_periods=2).std().fillna(0.0)
        std = _rename_frame(
            std_raw,
            [f"roll_std{seconds}s_{c}" for c in base_cols],
        )

        dev_values = log_res.values - mean_raw.values
        dev = _frame_from_values(
            dev_values,
            [f"dev_roll_mean{seconds}s_{c}" for c in base_cols],
            log_res.index,
        )

        features.extend([mean, std, dev])

    # 6) Sensor-type structure.
    type_to_indices = _sensor_type_index()
    type_mean_series = []
    type_std_series = []

    for typ, idxs in type_to_indices.items():
        cols = [f"logR_s{i + 1:02d}" for i in idxs]

        type_mean_series.append(
            log_res[cols].mean(axis=1).rename(f"type_mean_{typ}")
        )

        type_std_series.append(
            log_res[cols].std(axis=1).fillna(0.0).rename(f"type_std_{typ}")
        )

    type_mean = pd.concat(type_mean_series, axis=1)
    type_std = pd.concat(type_std_series, axis=1)

    type_d1_raw = type_mean.diff().fillna(0.0) / max(dt, eps)
    type_d1 = _rename_frame(
        type_d1_raw,
        [f"type_d1ps_{c}" for c in type_mean.columns],
    )

    features.extend([type_mean, type_std, type_d1])

    for seconds in type_slope_seconds:
        rows = _seconds_to_rows(seconds, dt, min_rows=2)

        type_lag_raw = type_mean.shift(rows)
        type_slope_values = (type_mean.values - type_lag_raw.values) / max(seconds, eps)

        type_slope = _frame_from_values(
            type_slope_values,
            [f"type_slope{seconds}s_{c}" for c in type_mean.columns],
            type_mean.index,
        )

        features.append(type_slope)

    # 7) Type contrasts and type interactions for cross-sensitivity.
    contrast_features = []
    type_cols = list(type_mean.columns)

    for i in range(len(type_cols)):
        for j in range(i + 1, len(type_cols)):
            a, b = type_cols[i], type_cols[j]

            contrast_features.append(
                (type_mean[a] - type_mean[b]).rename(f"type_contrast_{a}_minus_{b}")
            )

            contrast_features.append(
                (type_mean[a] * type_mean[b]).rename(f"type_interaction_{a}_x_{b}")
            )

    features.append(pd.concat(contrast_features, axis=1))

    # 8) Array-level summaries.
    array_stats = pd.DataFrame(
        {
            "array_mean": log_res.mean(axis=1),
            "array_std": log_res.std(axis=1).fillna(0.0),
            "array_min": log_res.min(axis=1),
            "array_max": log_res.max(axis=1),
            "array_range": log_res.max(axis=1) - log_res.min(axis=1),
            "array_d1_abs_mean": d1_raw.abs().mean(axis=1),
            "array_d2_abs_mean": d2_raw.abs().mean(axis=1),
            "array_rising_energy": d1_pos_raw.mean(axis=1),
            "array_recovery_energy": d1_neg_raw.mean(axis=1),
            "array_rise_recovery_balance": d1_pos_raw.mean(axis=1) - d1_neg_raw.mean(axis=1),
        },
        index=df.index,
    )

    features.append(array_stats)

    X = pd.concat(features, axis=1)
    X = _clean_features(X)

    return X, list(X.columns)


def make_transition_weights(
    y: np.ndarray,
    train_idx: np.ndarray,
    strength: float = 2.0,
    quantile: float = 0.90,
) -> np.ndarray:
    """
    Sample weights for sparse concentration transitions.

    Computed from normalized target changes using train statistics only.
    """
    mean = np.nanmean(y[train_idx], axis=0)
    std = np.nanstd(y[train_idx], axis=0)
    std = np.where(std < 1e-9, 1.0, std)

    y_norm = (y - mean) / std
    dy = np.abs(np.diff(y_norm, axis=0, prepend=y_norm[[0]])).mean(axis=1)

    threshold = float(np.quantile(dy[train_idx], quantile))
    if not np.isfinite(threshold) or threshold <= 1e-12:
        return np.ones(len(y), dtype=np.float32)

    score = np.clip(dy / threshold, 0.0, 3.0)
    weights = 1.0 + strength * score

    return weights.astype(np.float32)
