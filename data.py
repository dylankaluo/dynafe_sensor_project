from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


SENSOR_TYPES = [
    "TGS2602", "TGS2602", "TGS2600", "TGS2600",
    "TGS2610", "TGS2610", "TGS2620", "TGS2620",
    "TGS2602", "TGS2602", "TGS2600", "TGS2600",
    "TGS2610", "TGS2610", "TGS2620", "TGS2620",
]


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_dataset_file(data_dir: Path, mixture: str, must_exist: bool = True) -> Optional[Path]:
    """Same style as your old interface: recursively search the extracted UCI txt files."""
    data_dir = Path(data_dir)
    files = list(data_dir.rglob("*.txt"))

    if mixture == "ethylene_co":
        candidates = [
            p for p in files
            if "ethylene" in p.name.lower()
            and ("co" in p.name.lower() or "carbon" in p.name.lower())
            and "methane" not in p.name.lower()
        ]
    elif mixture == "ethylene_methane":
        candidates = [
            p for p in files
            if "ethylene" in p.name.lower() and "methane" in p.name.lower()
        ]
    else:
        raise ValueError(f"Unknown mixture: {mixture}")

    if candidates:
        return candidates[0]

    if must_exist:
        raise FileNotFoundError(
            f"Could not find data file for {mixture} under {data_dir}. "
            "Expected files similar to ethylene_CO.txt or ethylene_methane.txt."
        )
    return None


def read_sensor_file(
    file_path: Path,
    mixture: str,
    step: int,
    max_samples: Optional[int],
) -> Tuple[pd.DataFrame, List[str]]:
    """Read a UCI sensor file and keep your previous target naming convention."""
    target2_name = "ethylene_ppm"
    target1_name = "co_ppm" if mixture == "ethylene_co" else "methane_ppm"
    cols = ["time_s", target1_name, target2_name] + [f"s{i:02d}" for i in range(1, 17)]

    nrows = None
    if max_samples is not None:
        nrows = int(max_samples * max(1, step))

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        skiprows=1,
        names=cols,
        nrows=nrows,
        engine="python",
    )

    if step > 1:
        df = df.iloc[::step].reset_index(drop=True)

    if max_samples is not None:
        df = df.iloc[:max_samples].reset_index(drop=True)

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    return df, [target1_name, target2_name]


def make_synthetic_sensor_data(
    n: int = 5000,
    mixture: str = "ethylene_co",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
    """Optional offline sanity-check data. Real experiments should use the UCI files."""
    rng = np.random.default_rng(random_state)
    time_s = np.arange(n, dtype=float)

    target1_name = "co_ppm" if mixture == "ethylene_co" else "methane_ppm"
    target2_name = "ethylene_ppm"

    max_t1 = 600.0 if mixture == "ethylene_co" else 300.0
    max_t2 = 20.0

    y1 = np.zeros(n, dtype=float)
    y2 = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        length = int(rng.integers(60, 160))
        j = min(n, i + length)
        y1[i:j] = rng.choice([0.0, rng.uniform(0.05, 1.0) * max_t1])
        y2[i:j] = rng.choice([0.0, rng.uniform(0.05, 1.0) * max_t2])
        i = j

    sensors = np.zeros((n, 16), dtype=float)
    sensitivity = rng.normal(size=(16, 2))
    sensitivity[:, 0] *= 0.03
    sensitivity[:, 1] *= 0.90

    baseline = rng.uniform(8.0, 14.0, size=16)
    tau = rng.uniform(8.0, 55.0, size=16)
    alpha = 1.0 / tau
    state = baseline.copy()

    for t in range(n):
        conc = np.array([y1[t], y2[t]], dtype=float)
        interaction = 0.002 * np.sqrt(max(y1[t], 0.0)) * max(y2[t], 0.0)
        desired = baseline + sensitivity @ conc + interaction
        drift = 0.3 * np.sin(2 * np.pi * t / max(800, n // 3))
        noise = rng.normal(0.0, 0.08, size=16)
        state = state + alpha * (desired - state)
        sensors[t] = np.maximum(0.1, state + drift + noise)

    df = pd.DataFrame({"time_s": time_s, target1_name: y1, target2_name: y2})
    for k in range(16):
        df[f"s{k + 1:02d}"] = sensors[:, k]

    return df, [target1_name, target2_name]


def estimate_dt(df: pd.DataFrame) -> float:
    t = df["time_s"].astype(float).values
    if len(t) > 2:
        dt = float(np.nanmedian(np.diff(t)))
        if np.isfinite(dt) and dt > 0:
            return dt
    return 1.0


def load_dataset_from_args(args):
    """Shared loader used by train.py."""
    if getattr(args, "synthetic", False):
        df, target_cols = make_synthetic_sensor_data(
            n=args.max_samples or 5000,
            mixture=args.mixture,
            random_state=args.seed,
        )
        source = "synthetic"
    else:
        path = find_dataset_file(Path(args.data_dir), args.mixture, must_exist=True)
        df, target_cols = read_sensor_file(
            file_path=path,
            mixture=args.mixture,
            step=args.step,
            max_samples=args.max_samples,
        )
        source = str(path)

    y = df[target_cols].astype(float).values.astype(np.float32)
    time_s = df["time_s"].astype(float).values.astype(np.float32)
    dt = estimate_dt(df)

    return df, y, time_s, target_cols, dt, source


def chronological_split_indices(
    n_samples: int,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/val/test split."""
    split_test = int(n_samples * (1.0 - test_size))
    split_test = min(max(split_test, 20), n_samples - 10)

    train_val_idx = np.arange(0, split_test, dtype=np.int64)
    test_idx = np.arange(split_test, n_samples, dtype=np.int64)

    split_val = int(len(train_val_idx) * (1.0 - val_size))
    split_val = min(max(split_val, 10), len(train_val_idx) - 10)

    train_idx = train_val_idx[:split_val]
    val_idx = train_val_idx[split_val:]

    return train_idx, val_idx, test_idx
