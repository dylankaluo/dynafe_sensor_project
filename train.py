from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from data import chronological_split_indices, load_dataset_from_args, set_seed
from features import build_dynafe_features, make_transition_weights


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: float = 2.0, dropout: float = 0.10) -> None:
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DynaFENet(nn.Module):
    """
    Dynamic Feature Enhanced Neural Network.

    It takes causal engineered dynamic features and learns a nonlinear residual mapping
    using a lightweight residual MLP with target-specific heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.10,
        num_targets: int = 2,
    ) -> None:
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.blocks = nn.Sequential(*[
            ResidualBlock(hidden_dim, hidden_mult=2.0, dropout=dropout)
            for _ in range(num_blocks)
        ])

        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_targets)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input(x)
        h = self.blocks(h)
        h = self.shared(h)
        outs = [head(h) for head in self.heads]
        return torch.cat(outs, dim=1)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Dict[str, float]:
    report: Dict[str, float] = {}
    eps = 1e-9

    rmses, maes, r2s, smapes, nrmses = [], [], [], [], []
    for i, name in enumerate(target_names):
        rmse = math.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        smape = float(
            np.mean(
                2.0 * np.abs(y_pred[:, i] - y_true[:, i])
                / (np.abs(y_true[:, i]) + np.abs(y_pred[:, i]) + eps)
            ) * 100.0
        )
        y_range = float(np.nanmax(y_true[:, i]) - np.nanmin(y_true[:, i]))
        nrmse = float(rmse / max(y_range, eps))

        report[f"{name}_rmse"] = float(rmse)
        report[f"{name}_mae"] = float(mae)
        report[f"{name}_r2"] = float(r2)
        report[f"{name}_smape_pct"] = float(smape)
        report[f"{name}_nrmse_range"] = float(nrmse)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)
        smapes.append(smape)
        nrmses.append(nrmse)

    report["overall_rmse_mean"] = float(np.mean(rmses))
    report["overall_mae_mean"] = float(np.mean(maes))
    report["overall_r2_mean"] = float(np.mean(r2s))
    report["overall_smape_pct_mean"] = float(np.mean(smapes))
    report["overall_nrmse_range_mean"] = float(np.mean(nrmses))
    return report


def fit_target_scaler(y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(y_train, axis=0).astype(np.float32)
    std = np.nanstd(y_train, axis=0).astype(np.float32)
    std = np.where(std < 1e-9, 1.0, std).astype(np.float32)
    return mean, std


def scale_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((y - mean) / std).astype(np.float32)


def unscale_y(y_scaled: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return y_scaled * std + mean


def make_loader(
    X: np.ndarray,
    y_scaled: np.ndarray,
    weights: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X[indices].astype(np.float32)),
        torch.from_numpy(y_scaled[indices].astype(np.float32)),
        torch.from_numpy(weights[indices].astype(np.float32)),
        torch.from_numpy(indices.astype(np.int64)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    raw = F.huber_loss(pred, target, delta=delta, reduction="none").mean(dim=1)
    return (raw * sample_weight).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args,
    epoch: int,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_n = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch:03d}", dynamic_ncols=True, leave=False)
    for xb, yb, wb, _ in pbar:
        xb = xb.to(device)
        yb = yb.to(device)
        wb = wb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = weighted_huber_loss(pred, yb, wb, delta=args.huber_delta)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        n = xb.size(0)
        total_loss += float(loss.item()) * n
        total_n += n

        pbar.set_postfix({"loss": f"{total_loss / max(total_n, 1):.4f}"})

    return {"loss": total_loss / max(total_n, 1)}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    preds, targets, indices = [], [], []
    for xb, yb, _, idx in tqdm(loader, desc=desc, dynamic_ncols=True, leave=False):
        xb = xb.to(device)
        pred_scaled = model(xb).cpu().numpy()
        target_scaled = yb.numpy()

        preds.append(unscale_y(pred_scaled, y_mean, y_std))
        targets.append(unscale_y(target_scaled, y_mean, y_std))
        indices.append(idx.numpy())

    return np.concatenate(preds), np.concatenate(targets), np.concatenate(indices)


def plot_predictions(
    time_values: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: List[str],
    out_dir: Path,
    max_points: int = 1500,
) -> None:
    n = min(max_points, len(time_values))
    if n <= 1:
        return

    idx = np.linspace(0, len(time_values) - 1, n).astype(int)

    for i, target in enumerate(target_names):
        plt.figure(figsize=(10, 4))
        plt.plot(time_values[idx], y_true[idx, i], label="True", linewidth=1.5)
        plt.plot(time_values[idx], y_pred[idx, i], label="Predicted", linewidth=1.2)
        plt.xlabel("Time (s)")
        plt.ylabel(target)
        plt.title(f"DynaFE-Net Prediction vs True - {target}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"prediction_{target}.png", dpi=180)
        plt.close()


def run(args) -> Path:
    set_seed(args.seed)
    device = get_device(args.device)

    df, y, time_s, target_cols, dt, source = load_dataset_from_args(args)

    if len(df) < 200:
        raise ValueError("Too few samples. Increase --max-samples or lower --step.")

    out_dir = Path(args.results_dir) / f"dynafe_{args.mixture}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] Source: {source}")
    print(f"[run] Mixture: {args.mixture}")
    print(f"[run] Samples: {len(df)}")
    print(f"[run] Targets: {target_cols}")
    print(f"[run] dt: {dt:.4f}s")
    print(f"[run] Device: {device}")
    print(f"[run] Output: {out_dir}")

    print("[features] Building causal dynamic features...")
    X_df, feature_names = build_dynafe_features(df, dt=dt, mode=args.feature_mode)
    X_raw = X_df.values.astype(np.float32)
    print(f"[features] Feature mode: {args.feature_mode}")
    print(f"[features] Feature dimension: {X_raw.shape[1]}")

    train_idx, val_idx, test_idx = chronological_split_indices(
        n_samples=len(X_raw),
        test_size=args.test_size,
        val_size=args.val_size,
    )
    print(f"[split] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Fit scalers on train only.
    x_scaler = StandardScaler()
    x_scaler.fit(X_raw[train_idx])
    X = x_scaler.transform(X_raw).astype(np.float32)

    y_mean, y_std = fit_target_scaler(y[train_idx])
    y_scaled = scale_y(y, y_mean, y_std)

    sample_weights = make_transition_weights(
        y=y,
        train_idx=train_idx,
        strength=args.transition_weight,
        quantile=args.transition_quantile,
    )

    train_loader = make_loader(
        X=X,
        y_scaled=y_scaled,
        weights=sample_weights,
        indices=train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_loader(
        X=X,
        y_scaled=y_scaled,
        weights=sample_weights,
        indices=val_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        X=X,
        y_scaled=y_scaled,
        weights=sample_weights,
        indices=test_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = DynaFENet(
        input_dim=X.shape[1],
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
        num_targets=len(target_cols),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, args.patience // 2),
    )

    print(f"[model] DynaFE-Net input={X.shape[1]}, hidden={args.hidden_dim}, blocks={args.num_blocks}")

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    logs = []

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        train_log = train_one_epoch(model, train_loader, optimizer, device, args, epoch)

        val_pred, val_true, _ = predict(
            model=model,
            loader=val_loader,
            device=device,
            y_mean=y_mean,
            y_std=y_std,
            desc=f"Val Epoch {epoch:03d}",
        )
        val_metrics = regression_report(val_true, val_pred, target_cols)
        val_rmse = val_metrics["overall_rmse_mean"]

        scheduler.step(val_rmse)

        row = {
            "epoch": epoch,
            "train_loss": train_log["loss"],
            "val_rmse": val_rmse,
            "val_mae": val_metrics["overall_mae_mean"],
            "val_r2": val_metrics["overall_r2_mean"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        logs.append(row)

        epoch_bar.set_postfix({
            "loss": f"{train_log['loss']:.4f}",
            "val_rmse": f"{val_rmse:.4f}",
            "val_r2": f"{val_metrics['overall_r2_mean']:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        print(
            f"[epoch {epoch:03d}] "
            f"loss={train_log['loss']:.4f} "
            f"val_rmse={val_rmse:.4f} "
            f"val_r2={val_metrics['overall_r2_mean']:.4f}"
        )

        if val_rmse < best_val - 1e-6:
            best_val = val_rmse
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "epoch": epoch,
                "val_rmse": val_rmse,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"[early-stop] No improvement for {args.patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    test_pred, test_true, test_indices = predict(
        model=model,
        loader=test_loader,
        device=device,
        y_mean=y_mean,
        y_std=y_std,
        desc="Test",
    )
    metrics = regression_report(test_true, test_pred, target_cols)

    test_time = time_s[test_indices]
    pred_df = pd.DataFrame({"time_s": test_time})
    for i, name in enumerate(target_cols):
        pred_df[f"{name}_true"] = test_true[:, i]
        pred_df[f"{name}_pred"] = test_pred[:, i]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    summary = {
        "method": "DynaFE-Net",
        "full_name": "Dynamic Feature Enhanced Neural Network",
        "dataset_source": source,
        "mixture": args.mixture,
        "n_samples": int(len(df)),
        "targets": target_cols,
        "dt_seconds": float(dt),
        "feature_mode": args.feature_mode,
        "n_features": int(X.shape[1]),
        "args": vars(args),
        "target_mean": y_mean.tolist(),
        "target_std": y_std.tolist(),
        "best_val_rmse": float(best_val),
        "test_metrics": metrics,
        "design": {
            "log_resistance_transform": "scale-stable representation of positive sensor readings",
            "causal_dynamic_features": "lags, slopes, derivatives, EMA, rolling statistics",
            "sensor_type_features": "type mean/std/contrast/interaction for repeated sensor groups",
            "transition_weighted_loss": "higher loss weight around sparse concentration transitions",
            "residual_mlp": "lightweight neural network for nonlinear cross-sensitivity",
            "target_standardization": "balances gas concentration scales",
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_plots:
        plot_predictions(test_time, test_true, test_pred, target_cols, out_dir)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_names": feature_names,
            "x_scaler_mean": x_scaler.mean_,
            "x_scaler_scale": x_scaler.scale_,
            "target_mean": y_mean,
            "target_std": y_std,
            "target_cols": target_cols,
            "args": vars(args),
        },
        out_dir / "model.pt",
    )

    print("\n[done] Test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"[done] Results saved to: {out_dir}")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DynaFE-Net on dynamic chemical sensor data.")

    parser.add_argument("--mixture", default="ethylene_co", choices=["ethylene_co", "ethylene_methane"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic sanity-check data.")

    parser.add_argument("--preset", default="quick", choices=["quick", "full"])
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.20)

    parser.add_argument("--feature-mode", default=None, choices=["compact", "full"])

    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.10)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=1.5)
    parser.add_argument("--transition-quantile", type=float, default=0.90)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-plots", dest="save_plots", action="store_false")
    parser.set_defaults(save_plots=True)

    return parser


def apply_preset_defaults(args):
    if args.step is None:
        args.step = 100 if args.preset == "quick" else 20

    if args.max_samples is None:
        args.max_samples = 3000 if args.preset == "quick" else 100000

    if args.feature_mode is None:
        args.feature_mode = "compact" if args.preset == "quick" else "full"

    if args.hidden_dim is None:
        args.hidden_dim = 128 if args.preset == "quick" else 256

    if args.num_blocks is None:
        args.num_blocks = 2 if args.preset == "quick" else 3

    if args.epochs is None:
        args.epochs = 20 if args.preset == "quick" else 60

    if args.batch_size is None:
        args.batch_size = 256 if args.preset == "quick" else 1024

    if args.patience is None:
        args.patience = 6 if args.preset == "quick" else 10

    return args


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args = apply_preset_defaults(args)
    run(args)


if __name__ == "__main__":
    main()
