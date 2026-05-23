"""PRML Assignment 4 - Transformer module ablation study.

The experiment uses a synthetic pointer-retrieval task. The first token is a
query telling the model which later position to read, and the target is the
digit stored at that position. This keeps the experiment small enough for CPU
execution while stressing positional information, Q/K/V separation, residual
connections, and global self-attention.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
RESULT_DIR = os.path.join(BASE_DIR, "results")

RANDOM_STATE = 42
VALUE_VOCAB = 10
SEQ_LEN = 8
TOTAL_LEN = SEQ_LEN + 1
VOCAB_SIZE = VALUE_VOCAB + SEQ_LEN
QUERY_OFFSET = VALUE_VOCAB

TRAIN_STEPS = 700
BATCH_SIZE = 128
EVAL_BATCHES = 16
LEARNING_RATE = 1e-3
DEVICE = torch.device("cpu")


@dataclass
class ExperimentConfig:
    name: str
    model_type: str = "transformer"
    position_mode: str = "sinusoidal"
    tie_kv: bool = False
    residual_mode: str = "standard"
    layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.05


def ensure_dirs() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def generate_pointer_batch(
    batch_size: int,
    seq_len: int = SEQ_LEN,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create batches for the pointer-retrieval task.

    Example: query token 10+5 means "read the 6th value token"; the label is
    therefore values[:, 5].
    """

    query_index = torch.randint(0, seq_len, (batch_size,), generator=generator)
    values = torch.randint(0, VALUE_VOCAB, (batch_size, seq_len), generator=generator)
    query_token = (QUERY_OFFSET + query_index).unsqueeze(1)
    tokens = torch.cat([query_token, values], dim=1)
    labels = values[torch.arange(batch_size), query_index]
    return tokens.to(DEVICE), labels.to(DEVICE), query_index.to(DEVICE)


class PositionalEncoding(nn.Module):
    def __init__(self, mode: str, max_len: int, d_model: int) -> None:
        super().__init__()
        self.mode = mode
        self.d_model = d_model
        if mode == "learned":
            self.position = nn.Embedding(max_len, d_model)
        elif mode == "simple_abs":
            values = torch.linspace(-1.0, 1.0, steps=max_len).view(1, max_len, 1)
            self.register_buffer("simple_values", values)
        elif mode in {"sinusoidal", "adaptive_sinusoidal"}:
            pe = torch.zeros(max_len, d_model)
            positions = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(positions * div_term)
            pe[:, 1::2] = torch.cos(positions * div_term)
            self.register_buffer("sinusoidal", pe.unsqueeze(0))
            if mode == "adaptive_sinusoidal":
                self.position_scale = nn.Parameter(torch.ones(d_model))
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown position mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        if self.mode == "learned":
            positions = torch.arange(x.size(1), device=x.device)
            return x + self.position(positions).unsqueeze(0)
        if self.mode == "simple_abs":
            return x + self.simple_values[:, : x.size(1)].to(x.device).repeat(1, 1, self.d_model)
        pe = self.sinusoidal[:, : x.size(1)].to(x.device)
        if self.mode == "adaptive_sinusoidal":
            pe = pe * self.position_scale.view(1, 1, -1)
        return x + pe


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, tie_kv: bool = False) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.tie_kv = tie_kv
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = self.k_proj if tie_kv else nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, dim = x.shape
        x = x.view(bsz, length, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        self.last_attention = weights.detach().cpu()
        out = torch.matmul(self.dropout(weights), v)
        out = out.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.residual_mode = config.residual_mode
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(
            config.d_model,
            config.n_heads,
            config.dropout,
            tie_kv=config.tie_kv,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)
        if config.residual_mode == "gated":
            self.attn_gate = nn.Parameter(torch.full((config.d_model,), -1.0))
            self.ffn_gate = nn.Parameter(torch.full((config.d_model,), -1.0))

    def _merge(self, x: torch.Tensor, update: torch.Tensor, gate: nn.Parameter | None = None) -> torch.Tensor:
        if self.residual_mode == "none":
            return update
        if self.residual_mode == "gated":
            assert gate is not None
            return x + torch.sigmoid(gate).view(1, 1, -1) * update
        return x + update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_update = self.dropout(self.attn(self.attn_norm(x)))
        x = self._merge(x, attn_update, getattr(self, "attn_gate", None))
        ffn_update = self.dropout(self.ffn(self.ffn_norm(x)))
        x = self._merge(x, ffn_update, getattr(self, "ffn_gate", None))
        return x


class PointerTransformer(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position = PositionalEncoding(config.position_mode, TOTAL_LEN, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, VALUE_VOCAB)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) * math.sqrt(self.config.d_model)
        x = self.dropout(self.position(x))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.final_norm(x[:, 0]))

    def first_layer_attention(self) -> np.ndarray:
        attn = self.layers[0].attn.last_attention
        if attn is None:
            raise RuntimeError("Attention has not been computed yet")
        return attn[0].mean(0).numpy()


class PositionalCNN(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position = PositionalEncoding(config.position_mode, TOTAL_LEN, config.d_model)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(config.d_model, config.d_model, kernel_size=3, padding=1)
                for _ in range(config.layers + 3)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(config.d_model) for _ in self.convs])
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, VALUE_VOCAB)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim)
        x = self.position(x)
        for conv, norm in zip(self.convs, self.norms):
            update = conv(norm(x).transpose(1, 2)).transpose(1, 2)
            x = x + self.dropout(F.gelu(update))
        return self.head(x[:, 0])


def build_model(config: ExperimentConfig) -> nn.Module:
    if config.model_type == "cnn":
        return PositionalCNN(config).to(DEVICE)
    return PointerTransformer(config).to(DEVICE)


@torch.no_grad()
def evaluate(model: nn.Module, batches: int = EVAL_BATCHES, seed: int = 2026) -> Dict[str, float]:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for _ in range(batches):
        tokens, labels, _ = generate_pointer_batch(BATCH_SIZE, generator=generator)
        logits = model(tokens)
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * len(labels)
        total_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        total_examples += len(labels)
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one(config: ExperimentConfig) -> Tuple[nn.Module, List[Dict[str, float]]]:
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(RANDOM_STATE + 17)
    history: List[Dict[str, float]] = []

    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        tokens, labels, _ = generate_pointer_batch(BATCH_SIZE, generator=generator)
        logits = model(tokens)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % 20 == 0 or step == TRAIN_STEPS:
            metrics = evaluate(model, batches=8, seed=RANDOM_STATE + step)
            history.append(
                {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "eval_loss": metrics["loss"],
                    "eval_accuracy": metrics["accuracy"],
                }
            )
    return model, history


def experiment_configs() -> List[ExperimentConfig]:
    return [
        ExperimentConfig(name="Transformer-sinusoidal"),
        ExperimentConfig(name="Transformer-learned-pos", position_mode="learned"),
        ExperimentConfig(name="Transformer-simple-abs", position_mode="simple_abs"),
        ExperimentConfig(name="No-position-encoding", position_mode="none"),
        ExperimentConfig(name="Shared-KV-attention", tie_kv=True),
        ExperimentConfig(name="No-residual", residual_mode="none"),
        ExperimentConfig(name="Positional-CNN", model_type="cnn", position_mode="learned"),
        ExperimentConfig(name="Adaptive-PE-transformer", position_mode="adaptive_sinusoidal"),
    ]


def save_results(
    configs: Iterable[ExperimentConfig],
    histories: Dict[str, List[Dict[str, float]]],
    finals: Dict[str, Dict[str, float]],
) -> None:
    rows = []
    for config in configs:
        row = asdict(config)
        row.update(finals[config.name])
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    summary.to_csv(os.path.join(RESULT_DIR, "metrics.csv"), index=False, float_format="%.6f")
    with open(os.path.join(RESULT_DIR, "histories.json"), "w", encoding="utf-8") as fh:
        json.dump(histories, fh, indent=2)
    with open(os.path.join(RESULT_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "task": "pointer_retrieval",
                "sequence_length": SEQ_LEN,
                "train_steps": TRAIN_STEPS,
                "batch_size": BATCH_SIZE,
                "best_model": str(summary.iloc[0]["name"]),
                "best_accuracy": float(summary.iloc[0]["accuracy"]),
            },
            fh,
            indent=2,
        )


def plot_learning_curves(histories: Dict[str, List[Dict[str, float]]]) -> None:
    selected = [
        "Transformer-sinusoidal",
        "No-position-encoding",
        "Shared-KV-attention",
        "No-residual",
        "Positional-CNN",
        "Adaptive-PE-transformer",
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for name in selected:
        frame = pd.DataFrame(histories[name])
        ax.plot(frame["step"], frame["eval_accuracy"], marker="o", ms=3, label=name)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Pointer-retrieval learning curves")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw4_learning_curves.png"), dpi=170)
    plt.close(fig)


def plot_metric_bars(finals: Dict[str, Dict[str, float]]) -> None:
    frame = pd.DataFrame(finals).T.sort_values("accuracy")
    colors = ["#C44E52" if acc < 0.4 else "#8172B3" if acc < 0.8 else "#55A868" for acc in frame["accuracy"]]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(frame.index, frame["accuracy"], color=colors)
    ax.set_xlabel("Test accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_title("Final accuracy by architecture variant")
    for index, value in enumerate(frame["accuracy"]):
        ax.text(min(value + 0.015, 0.98), index, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw4_accuracy_comparison.png"), dpi=170)
    plt.close(fig)


def plot_position_effects(finals: Dict[str, Dict[str, float]]) -> None:
    names = [
        "Transformer-sinusoidal",
        "Transformer-learned-pos",
        "Transformer-simple-abs",
        "No-position-encoding",
    ]
    values = [finals[name]["accuracy"] for name in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(names, values, color=["#4C72B0", "#55A868", "#8172B3", "#C44E52"])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_title("Effect of positional information")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw4_position_ablation.png"), dpi=170)
    plt.close(fig)


@torch.no_grad()
def plot_attention_example(model: nn.Module) -> None:
    if not isinstance(model, PointerTransformer):
        return
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(123)
    tokens, labels, query_index = generate_pointer_batch(1, generator=generator)
    _ = model(tokens)
    attention = model.first_layer_attention()
    labels_for_axis = ["query"] + [str(i) for i in range(SEQ_LEN)]

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(attention, cmap="viridis", vmin=0.0, vmax=float(attention.max()))
    ax.set_xticks(range(TOTAL_LEN))
    ax.set_yticks(range(TOTAL_LEN))
    ax.set_xticklabels(labels_for_axis, rotation=90)
    ax.set_yticklabels(labels_for_axis)
    ax.set_title(f"First-layer mean attention, query reads position {int(query_index.item())}")
    ax.set_xlabel("Attended token")
    ax.set_ylabel("Query token position")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, "hw4_attention_heatmap.png"), dpi=170)
    plt.close(fig)


def run_experiment() -> None:
    ensure_dirs()
    set_seed()
    print("=" * 72)
    print("PRML Assignment 4: Transformer Module Ablation")
    print("=" * 72)
    print(f"Task: query token points to one of {SEQ_LEN} value tokens; class is the pointed digit.")
    print(f"Train steps: {TRAIN_STEPS} | batch size: {BATCH_SIZE} | device: {DEVICE}")

    configs = experiment_configs()
    histories: Dict[str, List[Dict[str, float]]] = {}
    finals: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, nn.Module] = {}

    for config in configs:
        print(f"\nTraining {config.name}...")
        set_seed(RANDOM_STATE)
        model, history = train_one(config)
        final_metrics = evaluate(model, batches=EVAL_BATCHES, seed=20260521)
        histories[config.name] = history
        finals[config.name] = final_metrics
        trained_models[config.name] = model
        print(
            f"  accuracy={final_metrics['accuracy']:.4f} "
            f"loss={final_metrics['loss']:.4f}"
        )

    save_results(configs, histories, finals)
    plot_learning_curves(histories)
    plot_metric_bars(finals)
    plot_position_effects(finals)
    plot_attention_example(trained_models["Transformer-sinusoidal"])

    metric_table = pd.DataFrame(finals).T.sort_values("accuracy", ascending=False)
    print("\nFinal test metrics:")
    print(metric_table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nFigures saved in: {FIGURE_DIR}")
    print(f"Results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    run_experiment()
