"""PRML Assignment 3 - Air quality one-hour forecasting.

This script frames the Beijing PM2.5 air-quality data as a supervised
time-series problem: given the weather and pollution values from the previous
24 hours, forecast PM2.5 concentration in the next hour.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "archive")
TRAIN_PATH = os.path.join(DATA_DIR, "LSTM-Multivariate_pollution.csv")
EXTERNAL_TEST_PATH = os.path.join(DATA_DIR, "pollution_test_data1.csv")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
RESULT_DIR = os.path.join(BASE_DIR, "results")

LOOKBACK_HOURS = 24
TEST_RATIO = 0.2
RANDOM_STATE = 42
LSTM_EPOCHS = 35
LSTM_BATCH_SIZE = 256
LSTM_PATIENCE = 7

NUMERIC_COLUMNS = ["pollution", "dew", "temp", "press", "wnd_spd", "snow", "rain"]
CATEGORICAL_COLUMN = "wnd_dir"
TARGET_COLUMN = "pollution"

torch.manual_seed(RANDOM_STATE)
torch.set_num_threads(2)


@dataclass
class Preprocessor:
    """Minimal preprocessing state for numeric scaling and wind one-hot coding."""

    numeric_columns: list[str]
    wind_categories: list[str]
    means: pd.Series
    stds: pd.Series

    @property
    def feature_columns(self) -> list[str]:
        wind_columns = [f"{CATEGORICAL_COLUMN}_{cat}" for cat in self.wind_categories]
        return self.numeric_columns + wind_columns

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        numeric = frame[self.numeric_columns].copy()
        numeric = (numeric - self.means) / self.stds
        wind = pd.get_dummies(frame[CATEGORICAL_COLUMN], prefix=CATEGORICAL_COLUMN)
        wind = wind.reindex(
            columns=[f"{CATEGORICAL_COLUMN}_{cat}" for cat in self.wind_categories],
            fill_value=0,
        )
        features = pd.concat([numeric, wind], axis=1)
        return features[self.feature_columns].to_numpy(dtype=np.float64)


class LSTMForecaster(nn.Module):
    """Small CPU-friendly LSTM for one-hour PM2.5 forecasting."""

    def __init__(self, input_size: int, hidden_size: int = 48) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_step = output[:, -1, :]
        return self.head(last_step).squeeze(-1)


def ensure_dirs() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_main_data() -> pd.DataFrame:
    frame = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].apply(pd.to_numeric)
    return frame


def load_external_test_data() -> pd.DataFrame:
    frame = pd.read_csv(EXTERNAL_TEST_PATH)
    frame[NUMERIC_COLUMNS] = frame[NUMERIC_COLUMNS].apply(pd.to_numeric)
    return frame


def fit_preprocessor(train_frame: pd.DataFrame) -> Preprocessor:
    means = train_frame[NUMERIC_COLUMNS].mean()
    stds = train_frame[NUMERIC_COLUMNS].std().replace(0, 1.0)
    wind_categories = sorted(train_frame[CATEGORICAL_COLUMN].dropna().unique().tolist())
    return Preprocessor(NUMERIC_COLUMNS, wind_categories, means, stds)


def make_supervised(
    transformed_values: np.ndarray,
    pollution_values: np.ndarray,
    lookback: int = LOOKBACK_HOURS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build flattened lag windows and matching one-step-ahead targets."""

    n_rows = transformed_values.shape[0]
    rows = []
    for target_index in range(lookback, n_rows):
        rows.append(transformed_values[target_index - lookback : target_index].ravel())

    x = np.asarray(rows, dtype=np.float64)
    y = pollution_values[lookback:].astype(np.float64)
    target_indices = np.arange(lookback, n_rows)
    return x, y, target_indices


def split_supervised(
    x: np.ndarray,
    y: np.ndarray,
    target_indices: np.ndarray,
    split_row: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = target_indices < split_row
    test_mask = ~train_mask
    return (
        x[train_mask],
        x[test_mask],
        y[train_mask],
        y[test_mask],
        target_indices[train_mask],
        target_indices[test_mask],
    )


def build_models() -> Dict[str, object]:
    return {
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=40,
            max_depth=16,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=90,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.75,
            random_state=RANDOM_STATE,
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(80, 32),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=180,
            early_stopping=True,
            n_iter_no_change=12,
            random_state=RANDOM_STATE,
        ),
    }


def train_lstm_model(
    x_train_seq: np.ndarray,
    y_train: np.ndarray,
    x_test_seq: np.ndarray,
) -> Tuple[np.ndarray, LSTMForecaster, Dict[str, float], list[Dict[str, float]]]:
    """Train the PyTorch LSTM and return test predictions plus training history."""

    device = torch.device("cpu")
    validation_size = max(int(len(x_train_seq) * 0.15), 1)
    train_size = len(x_train_seq) - validation_size

    x_fit = x_train_seq[:train_size].astype(np.float32)
    y_fit = y_train[:train_size].astype(np.float32)
    x_val = x_train_seq[train_size:].astype(np.float32)
    y_val = y_train[train_size:].astype(np.float32)

    target_mean = float(y_fit.mean())
    target_std = float(y_fit.std() if y_fit.std() > 0 else 1.0)
    y_fit_scaled = (y_fit - target_mean) / target_std
    y_val_scaled = (y_val - target_mean) / target_std

    model = LSTMForecaster(input_size=x_train_seq.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    x_fit_tensor = torch.from_numpy(x_fit).to(device)
    y_fit_tensor = torch.from_numpy(y_fit_scaled).to(device)
    x_val_tensor = torch.from_numpy(x_val).to(device)
    y_val_tensor = torch.from_numpy(y_val_scaled).to(device)

    best_state = None
    best_val = float("inf")
    epochs_without_improvement = 0
    history: list[Dict[str, float]] = []

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        permutation = torch.randperm(train_size)
        batch_losses = []

        for start in range(0, train_size, LSTM_BATCH_SIZE):
            batch_idx = permutation[start : start + LSTM_BATCH_SIZE]
            xb = x_fit_tensor[batch_idx]
            yb = y_fit_tensor[batch_idx]

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val_tensor), y_val_tensor).item())

        train_loss = float(np.mean(batch_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"  LSTM epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= LSTM_PATIENCE:
                print(f"  LSTM early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    predictions = predict_lstm(model, x_test_seq, target_mean, target_std)
    target_stats = {"mean": target_mean, "std": target_std}
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(os.path.join(RESULT_DIR, "lstm_training_history.csv"), index=False)
    return predictions, model, target_stats, history


def predict_lstm(
    model: LSTMForecaster,
    x_seq: np.ndarray,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_seq), LSTM_BATCH_SIZE):
            batch = torch.from_numpy(x_seq[start : start + LSTM_BATCH_SIZE].astype(np.float32))
            scaled_pred = model(batch).cpu().numpy()
            predictions.append(scaled_pred * target_std + target_mean)
    return np.maximum(np.concatenate(predictions), 0.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    table = pd.DataFrame(metrics).T
    table = table[["MAE", "RMSE", "R2"]].sort_values("RMSE")
    table.to_csv(os.path.join(RESULT_DIR, "metrics.csv"), float_format="%.6f")
    with open(os.path.join(RESULT_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def plot_data_overview(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    first_month = frame.iloc[: 24 * 30]
    axes[0, 0].plot(first_month["date"], first_month["pollution"], color="#2F6B9A", lw=1.2)
    axes[0, 0].set_title("PM2.5 concentration during first 30 days")
    axes[0, 0].set_ylabel("PM2.5")
    axes[0, 0].tick_params(axis="x", rotation=30)

    hourly = frame.assign(hour=frame["date"].dt.hour).groupby("hour")["pollution"].mean()
    axes[0, 1].bar(hourly.index, hourly.values, color="#55A868")
    axes[0, 1].set_title("Average pollution by hour of day")
    axes[0, 1].set_xlabel("Hour")
    axes[0, 1].set_ylabel("Mean PM2.5")

    monthly = frame.assign(month=frame["date"].dt.month).groupby("month")["pollution"].mean()
    axes[1, 0].plot(monthly.index, monthly.values, marker="o", color="#C44E52")
    axes[1, 0].set_title("Average pollution by month")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Mean PM2.5")
    axes[1, 0].set_xticks(range(1, 13))

    wind_counts = frame[CATEGORICAL_COLUMN].value_counts().sort_index()
    axes[1, 1].bar(wind_counts.index, wind_counts.values, color="#8172B3")
    axes[1, 1].set_title("Wind direction distribution")
    axes[1, 1].set_xlabel("Wind direction")
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw3_data_overview.png"), dpi=160)
    plt.close(fig)


def plot_correlation(frame: pd.DataFrame) -> None:
    corr = frame[NUMERIC_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS)

    for i in range(len(NUMERIC_COLUMNS)):
        for j in range(len(NUMERIC_COLUMNS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=ax, shrink=0.85)
    ax.set_title("Correlation matrix of numeric variables")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw3_correlation.png"), dpi=160)
    plt.close(fig)


def plot_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    table = pd.DataFrame(metrics).T.loc[:, ["MAE", "RMSE", "R2"]]
    ordered = table.sort_values("RMSE")
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        values = ordered[metric].values
        ax.bar(ordered.index, values, color=colors[: len(ordered)])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Forecasting performance on the temporal hold-out set", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw3_metrics_comparison.png"), dpi=160)
    plt.close(fig)


def plot_lstm_training(history: list[Dict[str, float]]) -> None:
    if not history:
        return

    history_frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(history_frame["epoch"], history_frame["train_loss"], marker="o", label="Train loss")
    ax.plot(history_frame["epoch"], history_frame["val_loss"], marker="o", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Scaled MSE loss")
    ax.set_title("LSTM training curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw3_lstm_training.png"), dpi=160)
    plt.close(fig)


def plot_predictions(
    frame: pd.DataFrame,
    test_indices: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    best_model_name: str,
) -> None:
    sample_size = min(240, len(y_true))
    sample_indices = test_indices[:sample_size]
    dates = frame.loc[sample_indices, "date"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, y_true[:sample_size], label="Actual", color="#222222", lw=1.5)
    ax.plot(
        dates,
        predictions[best_model_name][:sample_size],
        label=best_model_name,
        color="#4C72B0",
        lw=1.4,
    )
    ax.plot(
        dates,
        predictions["Persistence"][:sample_size],
        label="Persistence",
        color="#C44E52",
        lw=1.0,
        alpha=0.8,
    )
    ax.set_title("First 240 test hours: actual vs forecast")
    ax.set_ylabel("PM2.5")
    ax.legend()
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw3_prediction_series.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    y_pred = predictions[best_model_name]
    ax.scatter(y_true, y_pred, s=10, alpha=0.35, color="#4C72B0")
    limit = max(float(np.max(y_true)), float(np.max(y_pred))) * 1.05
    ax.plot([0, limit], [0, limit], color="#C44E52", lw=1.2, ls="--")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Actual PM2.5")
    ax.set_ylabel("Predicted PM2.5")
    ax.set_title(f"Actual vs predicted ({best_model_name})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw3_prediction_scatter.png"), dpi=160)
    plt.close(fig)


def iter_lag_feature_names(base_features: Iterable[str], lookback: int) -> Iterable[Tuple[str, int]]:
    base_features = list(base_features)
    for lag_from_start in range(lookback):
        lag = lookback - lag_from_start
        for feature in base_features:
            yield feature, lag


def plot_feature_importance(
    model: RandomForestRegressor,
    base_feature_names: list[str],
    lookback: int = LOOKBACK_HOURS,
) -> None:
    importances = model.feature_importances_
    names = list(iter_lag_feature_names(base_feature_names, lookback))
    importance_frame = pd.DataFrame(
        {
            "feature": [name for name, _ in names],
            "lag": [lag for _, lag in names],
            "importance": importances,
        }
    )

    by_feature = importance_frame.groupby("feature")["importance"].sum().sort_values(ascending=False)
    by_lag = importance_frame.groupby("lag")["importance"].sum().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(by_feature.index, by_feature.values, color="#4C72B0")
    axes[0].set_title("Random Forest importance by variable")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Total importance")

    axes[1].plot(by_lag.index, by_lag.values, marker="o", color="#55A868")
    axes[1].invert_xaxis()
    axes[1].set_title("Random Forest importance by lag")
    axes[1].set_xlabel("Hours before target")
    axes[1].set_ylabel("Total importance")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw3_feature_importance.png"), dpi=160)
    plt.close(fig)


def run_experiment() -> None:
    ensure_dirs()
    print("=" * 72)
    print("PRML Assignment 3: Air Quality One-Hour Forecasting")
    print("=" * 72)

    frame = load_main_data()
    external_frame = load_external_test_data()
    split_row = int(len(frame) * (1.0 - TEST_RATIO))

    train_frame = frame.iloc[:split_row].copy()
    preprocessor = fit_preprocessor(train_frame)
    transformed = preprocessor.transform(frame)
    x, y, target_indices = make_supervised(
        transformed,
        frame[TARGET_COLUMN].to_numpy(dtype=np.float64),
    )
    x_train, x_test, y_train, y_test, train_indices, test_indices = split_supervised(
        x, y, target_indices, split_row
    )

    print(f"Rows: {len(frame)} | Train rows: {split_row} | Test rows: {len(frame) - split_row}")
    print(f"Supervised train samples: {len(x_train)} | test samples: {len(x_test)}")
    print(f"Lookback window: {LOOKBACK_HOURS} hours | Features per hour: {len(preprocessor.feature_columns)}")

    plot_data_overview(frame)
    plot_correlation(frame)

    predictions: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    persistence_pred = frame.loc[test_indices - 1, TARGET_COLUMN].to_numpy(dtype=np.float64)
    predictions["Persistence"] = persistence_pred
    metrics["Persistence"] = evaluate_predictions(y_test, persistence_pred)

    trained_models: Dict[str, object] = {}
    for name, model in build_models().items():
        print(f"Training {name}...")
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        y_pred = np.maximum(y_pred, 0.0)
        predictions[name] = y_pred
        metrics[name] = evaluate_predictions(y_test, y_pred)
        trained_models[name] = model

    n_base_features = len(preprocessor.feature_columns)
    x_train_seq = x_train.reshape(len(x_train), LOOKBACK_HOURS, n_base_features)
    x_test_seq = x_test.reshape(len(x_test), LOOKBACK_HOURS, n_base_features)
    print("Training LSTM...")
    lstm_pred, lstm_model, lstm_target_stats, lstm_history = train_lstm_model(
        x_train_seq,
        y_train,
        x_test_seq,
    )
    predictions["LSTM"] = lstm_pred
    metrics["LSTM"] = evaluate_predictions(y_test, lstm_pred)
    plot_lstm_training(lstm_history)

    save_metrics(metrics)

    metric_table = pd.DataFrame(metrics).T.loc[:, ["MAE", "RMSE", "R2"]].sort_values("RMSE")
    print("\nTemporal hold-out metrics:")
    print(metric_table.to_string(float_format=lambda v: f"{v:.4f}"))

    best_model_name = metric_table.index[0]
    plot_metrics(metrics)
    plot_predictions(frame, test_indices, y_test, predictions, best_model_name)
    if "Random Forest" in trained_models:
        plot_feature_importance(
            trained_models["Random Forest"],
            preprocessor.feature_columns,
        )

    external_results = {}
    if len(external_frame) > LOOKBACK_HOURS:
        external_transformed = preprocessor.transform(external_frame)
        x_external, y_external, external_indices = make_supervised(
            external_transformed,
            external_frame[TARGET_COLUMN].to_numpy(dtype=np.float64),
        )
        for name, model in trained_models.items():
            y_external_pred = np.maximum(model.predict(x_external), 0.0)
            external_results[name] = evaluate_predictions(y_external, y_external_pred)
        x_external_seq = x_external.reshape(len(x_external), LOOKBACK_HOURS, n_base_features)
        y_external_lstm = predict_lstm(
            lstm_model,
            x_external_seq,
            lstm_target_stats["mean"],
            lstm_target_stats["std"],
        )
        external_results["LSTM"] = evaluate_predictions(y_external, y_external_lstm)
        external_results["Persistence"] = evaluate_predictions(
            y_external,
            external_frame.loc[external_indices - 1, TARGET_COLUMN].to_numpy(dtype=np.float64),
        )
        external_table = pd.DataFrame(external_results).T[["MAE", "RMSE", "R2"]].sort_values("RMSE")
        external_table.to_csv(os.path.join(RESULT_DIR, "external_test_metrics.csv"), float_format="%.6f")
        print("\nExternal file metrics:")
        print(external_table.to_string(float_format=lambda v: f"{v:.4f}"))

    summary = {
        "lookback_hours": LOOKBACK_HOURS,
        "train_rows": int(split_row),
        "test_rows": int(len(frame) - split_row),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "best_model": str(best_model_name),
        "best_metrics": metrics[str(best_model_name)],
    }
    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nFigures saved in: {FIGURE_DIR}")
    print(f"Results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    run_experiment()
