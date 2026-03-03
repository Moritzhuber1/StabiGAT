#!/usr/bin/env python3
"""
A small, clean experiment framework to compare 3 cross-validation strategies:

1) StratifiedShuffleSplit (SSS)
2) StratifiedKFold (SKF)
3) GroupKFold (GKF) with clusters from mean-pooled ESM2 embeddings


Note:
- This script assumes your data layout is similar to your current project:
    data/raw/final.csv
    data/esm_embeddings/<name>.npy
    data/adjacency/<name>_adjacency_matrix.npy

How to run:
    python diff_cv.py

You can also import and call run_all_experiments() from another script.
"""

from __future__ import annotations

import json
import csv
import time
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import product
from typing import Dict, Iterable, Iterator, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Dropout
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt


# =============================================================================
# Global settings (keep this small; real configs go into dataclasses below)
# =============================================================================

ROOT = Path.cwd()
DATA_DIR = (ROOT / "../data").resolve()
EMB_PATH = DATA_DIR / "esm_embeddings"
ADJ_PATH = DATA_DIR / "adjacency"
CSV_PATH = DATA_DIR / "raw" / "final.csv"

RUNS_DIR = ROOT / "runs"
RUNS_DIR.mkdir(exist_ok=True)

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIM = 1280
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training defaults
BATCH_SIZE = 1
EPOCHS = 25
LEARNING_RATE = 1e-4
PATIENCE = 5


# =============================================================================
# Config objects (small, explicit, serializable)
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Model hyper-parameters."""
    hidden_dim: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.3


@dataclass(frozen=True)
class DataConfig:
    """Data / graph parameters."""
    cut_off: float = 0.5  # adjacency threshold


@dataclass(frozen=True)
class CVConfig:
    """
    CV strategy parameters.

    method:
      - "sss": StratifiedShuffleSplit
      - "skf": StratifiedKFold
      - "gkf": GroupKFold (needs n_clusters)
    """
    method: str = "skf"
    n_splits: int = 7
    test_size: float = 0.2  # only used for SSS
    seed: int = 42
    n_clusters: Optional[int] = None  # only used for GKF


@dataclass(frozen=True)
class ExperimentConfig:
    """Full experiment definition. This is one run folder."""
    data: DataConfig
    model: ModelConfig
    cv: CVConfig

    def tag(self) -> str:
        """Folder name for this run."""
        base = (
            f"cut{self.data.cut_off}_layers{self.model.num_layers}_hid{self.model.hidden_dim}"
            f"_drop{self.model.dropout_rate}_cv{self.cv.method}_splits{self.cv.n_splits}"
        )
        if self.cv.method == "gkf":
            base += "_mmseqs"  # indicate that groups are from MMseqs, not random clusters
        return base


# =============================================================================
# Model (same as your GATI with cross-attention + GAT stack)
# =============================================================================

class CrossAttention(nn.Module):
    """Single-head cross-attention: queries = nodes, keys/values = molecule feature."""

    def __init__(self, node_dim: int, md_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(node_dim, hidden_dim)
        self.key_proj = nn.Linear(md_dim, hidden_dim)
        self.value_proj = nn.Linear(md_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, node_dim)

    def forward(self, node_feats: torch.Tensor, md_feat: torch.Tensor) -> torch.Tensor:
        Q = self.query_proj(node_feats)                 # (L, H)
        K = self.key_proj(md_feat).unsqueeze(0)         # (1, H)
        V = self.value_proj(md_feat).unsqueeze(0)       # (1, H)

        attn_scores = (Q @ K.T) / (Q.shape[-1] ** 0.5)  # (L, 1)
        attn_weights = torch.softmax(attn_scores, dim=0) # (L, 1)

        attended = attn_weights * V                     # (L, H)
        return self.out_proj(attended) + node_feats      # residual


class GATI(nn.Module):
    """Graph-attention network with configurable depth/width."""

    def __init__(self, md_dim: int, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.num_layers >= 1, "Need at least one GAT layer"

        self.cross_attn = CrossAttention(
            node_dim=EMB_DIM,
            md_dim=md_dim,
            hidden_dim=cfg.hidden_dim,
        )

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.drop_layers = nn.ModuleList()

        in_channels = EMB_DIM
        for layer_idx in range(cfg.num_layers):
            heads = 4 if layer_idx < cfg.num_layers - 1 else 1
            gat = GATConv(in_channels, cfg.hidden_dim, heads=heads, edge_dim=1)
            self.gat_layers.append(gat)
            self.norm_layers.append(LayerNorm(cfg.hidden_dim * heads))
            self.drop_layers.append(Dropout(cfg.dropout_rate))
            in_channels = cfg.hidden_dim * heads

        self.cls = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout_rate),
            nn.Linear(in_channels // 2, 2),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x[:, :EMB_DIM], data.edge_index, data.edge_attr
        md_feat = data.x[0, EMB_DIM:]  # (md_dim,)

        x = self.cross_attn(x, md_feat)

        for gat, norm, drop in zip(self.gat_layers, self.norm_layers, self.drop_layers):
            x = gat(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = drop(x)

        graph_repr = x.mean(dim=0, keepdim=True)
        return self.cls(graph_repr)


# =============================================================================
# Data access (Single Responsibility: reading metadata and building graphs)
# =============================================================================

class MetadataRepository:
    """Loads CSV and returns df + raw features + labels."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path

    def load(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
        df = pd.read_csv(self._csv_path)
        df["label"] = (df["class"] == "stable").astype(int)

        feature_cols = [c for c in df.columns if c not in {"name", "sequence", "melting_point", "class"}]
        md_raw = df[feature_cols].values.astype(np.float32)
        labels = df["label"].values.astype(int)

        return df, md_raw, labels, feature_cols


class GraphBuilder:
    """
    Builds PyG graphs for given sample indices.

    Important:
    - It receives a fitted scaler (so we can do fold-wise scaling).
    - It is cut_off-dependent because edges depend on adjacency threshold.
    """

    def __init__(self, emb_path: Path, adj_path: Path, cut_off: float) -> None:
        self._emb_path = emb_path
        self._adj_path = adj_path
        self._cut_off = cut_off

    def _load_graph(self, name: str, md_feat_scaled: np.ndarray) -> Data:
        A = np.load(self._adj_path / f"{name}_adjacency_matrix.npy")
        X = np.load(self._emb_path / f"{name}.npy")
        if A.shape[0] != X.shape[0]:
            raise ValueError(f"Adjacency and embedding length mismatch for {name}")

        mask = A > self._cut_off
        edge_index = np.array(np.nonzero(mask))          # (2, E)
        edge_weight = A[mask].astype(np.float32)         # (E,)

        md_repeated = np.tile(md_feat_scaled, (X.shape[0], 1))
        X_combined = np.concatenate([X, md_repeated], axis=1)

        return Data(
            x=torch.tensor(X_combined, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_weight, dtype=torch.float32).unsqueeze(1),
        )

    def build(
        self,
        df: pd.DataFrame,
        md_raw: np.ndarray,
        indices: np.ndarray,
        scaler: StandardScaler,
    ) -> List[Data]:
        names = df["name"].values
        labels = df["label"].values.astype(int)

        md_scaled = scaler.transform(md_raw[indices])
        out: List[Data] = []

        for pos, i in enumerate(indices):
            g = self._load_graph(names[i], md_scaled[pos])
            g.y = torch.tensor([int(labels[i])], dtype=torch.long)
            out.append(g)

        return out


# =============================================================================
# Group / cluster building (Single Responsibility + caching)
# =============================================================================

class GroupAssigner:
    """Builds groups from mean-pooled ESM embeddings and caches results."""

    def __init__(self, emb_path: Path, cache_dir: Path) -> None:
        self._emb_path = emb_path
        self._cache_dir = cache_dir
        self._mem_cache: Dict[int, np.ndarray] = {}

    def _mean_pool_esm(self, name: str) -> np.ndarray:
        X = np.load(self._emb_path / f"{name}.npy")  # (L, EMB_DIM)
        return X.mean(axis=0)

    def get_groups(self, df: pd.DataFrame, n_clusters: int) -> np.ndarray:
        # 1) In-memory cache (fastest)
        if n_clusters in self._mem_cache:
            return self._mem_cache[n_clusters]

        # 2) Disk cache (useful across runs)
        cache_file = self._cache_dir / f"groups_clus{n_clusters}.npy"
        if cache_file.exists():
            groups = np.load(cache_file).astype(int)
            self._mem_cache[n_clusters] = groups
            return groups

        # 3) Compute and store
        pooled = np.vstack(
            [self._mean_pool_esm(nm) for nm in tqdm(df["name"], desc=f"Mean-pool ESM (clus={n_clusters})")]
        )
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        groups = clustering.fit_predict(pooled).astype(int)

        np.save(cache_file, groups)
        self._mem_cache[n_clusters] = groups
        return groups


# =============================================================================
# CV splitting (Open/Closed: add a new splitter class without changing trainer)
# =============================================================================

class SplitStrategy(Protocol):
    """Interface for split strategies (SOLID: dependency inversion)."""

    def splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Yields: (fold_id, train_indices, test_indices)
        fold_id starts at 1.
        """
        ...


class StratifiedShuffleSplitStrategy:
    """StratifiedShuffleSplit with n_splits folds and fixed test_size."""

    def __init__(self, n_splits: int, test_size: float, seed: int) -> None:
        self._sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

    def splits(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        for fold_id, (tr, te) in enumerate(self._sss.split(X, y), start=1):
            yield fold_id, tr, te


class StratifiedKFoldStrategy:
    """StratifiedKFold for deterministic k-fold evaluation."""

    def __init__(self, n_splits: int, seed: int) -> None:
        self._skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def splits(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        for fold_id, (tr, te) in enumerate(self._skf.split(X, y), start=1):
            yield fold_id, tr, te


class GroupKFoldStrategy:
    """GroupKFold. Needs groups array."""

    def __init__(self, n_splits: int) -> None:
        self._gkf = GroupKFold(n_splits=n_splits)

    def splits(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        if groups is None:
            raise ValueError("GroupKFold needs groups, but groups=None was given.")
        for fold_id, (tr, te) in enumerate(self._gkf.split(X, y, groups=groups), start=1):
            yield fold_id, tr, te


def make_split_strategy(cv_cfg: CVConfig) -> SplitStrategy:
    """Factory function to select a strategy by config."""
    if cv_cfg.method == "sss":
        return StratifiedShuffleSplitStrategy(cv_cfg.n_splits, cv_cfg.test_size, cv_cfg.seed)
    if cv_cfg.method == "skf":
        return StratifiedKFoldStrategy(cv_cfg.n_splits, cv_cfg.seed)
    if cv_cfg.method == "gkf":
        return GroupKFoldStrategy(cv_cfg.n_splits)
    raise ValueError(f"Unknown cv.method: {cv_cfg.method}")


# =============================================================================
# Training and evaluation (Single Responsibility: learn + score)
# =============================================================================

@dataclass
class FoldMetrics:
    fold: int
    accuracy: float
    f1: float
    roc_auc: float
    pr_auc: float


class Trainer:
    """Trains one fold with early stopping and returns best model state."""

    def __init__(self, lr: float = LEARNING_RATE, epochs: int = EPOCHS, patience: int = PATIENCE) -> None:
        self._lr = lr
        self._epochs = epochs
        self._patience = patience

    def _train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        opt: torch.optim.Optimizer,
        class_weights: torch.Tensor,
    ) -> float:
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y, weight=class_weights)

            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += float(loss.item())

        return total_loss / max(1, len(loader))

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Tuple[float, float, float, float, float, List[int], List[int]]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        y_true: List[int] = []
        y_pred: List[int] = []
        y_prob: List[float] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(DEVICE)
                logits = model(batch)

                probs = torch.softmax(logits, dim=1)[:, 1]
                loss = F.cross_entropy(logits, batch.y)

                pred = logits.argmax(dim=1)
                correct += int((pred == batch.y).sum().item())
                total += int(len(batch.y))
                total_loss += float(loss.item())

                y_true.extend(batch.y.cpu().numpy().tolist())
                y_pred.extend(pred.cpu().numpy().tolist())
                y_prob.extend(probs.cpu().numpy().tolist())

        acc = correct / max(1, total)
        f1 = f1_score(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")

        try:
            roc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
            prc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
        except ValueError:
            roc = float("nan")
            prc = float("nan")

        return acc, total_loss / max(1, len(loader)), f1, roc, prc, y_true, y_pred

    def train_fold(
        self,
        md_dim: int,
        model_cfg: ModelConfig,
        train_graphs: List[Data],
        val_graphs: List[Data],
        test_graphs: List[Data],
        out_dir: Path,
        fold_id: int,
    ) -> Tuple[Dict[str, object], Dict[str, float], Dict[str, object]]:
        
        """
        Returns:
          - best_state_dict
          - fold_summary metrics
          - training history dict (for CSV)
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        tr_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        va_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
        te_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

        model = GATI(md_dim=md_dim, cfg=model_cfg).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=self._lr)

        # You can adapt weights if the dataset is imbalanced.
        class_weights = torch.tensor([1.0, 1.0], device=DEVICE)

        best_f1 = -1.0
        best_state: Optional[Dict[str, object]] = None
        patience = 0
        history_rows: List[Dict[str, float]] = []

        for epoch in range(1, self._epochs + 1):
            tr_loss = self._train_epoch(model, tr_loader, opt, class_weights)
            val_acc, val_loss, val_f1, val_roc, val_prc, _, _ = self._evaluate(model, va_loader)

            history_rows.append(
                dict(
                    epoch=epoch,
                    train_loss=tr_loss,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    val_f1=val_f1,
                    roc_auc=val_roc,
                    pr_auc=val_prc,
                )
            )

            # Early stopping by validation F1
            if (not np.isnan(val_f1)) and (val_f1 > best_f1):
                best_f1 = val_f1
                best_state = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= self._patience:
                    break

        if best_state is None:
            best_state = copy.deepcopy(model.state_dict())

        # Save learning curve
        pd.DataFrame(history_rows).to_csv(out_dir / f"fold{fold_id}_learning.csv", index=False)

        # Evaluate with best weights and create artifacts
        model.load_state_dict(best_state)
        acc, _, f1, roc, prc, y_true, y_pred = self._evaluate(model, te_loader)

        # Save fold model
        torch.save(best_state, out_dir / f"model_fold{fold_id}.pt")

        # Save fold metrics (CSV with 1 row)
        fold_row = dict(fold=fold_id, accuracy=acc, f1=f1, roc_auc=roc, pr_auc=prc)
        with open(out_dir / f"fold{fold_id}_metrics.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fold_row.keys()))
            writer.writeheader()
            writer.writerow(fold_row)

        # Confusion matrix
        self._plot_cm(y_true, y_pred, out_dir / f"fold{fold_id}_cm.png")

        return best_state, fold_row, {"history": history_rows}

    @staticmethod
    def _plot_cm(y_true: List[int], y_pred: List[int], out_path: Path) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["unstable", "stable"])
        disp.plot(cmap="Blues")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()


# =============================================================================
# Experiment runner (Single Responsibility: orchestrate one config)
# =============================================================================

class ExperimentRunner:
    """Runs one ExperimentConfig end-to-end and writes artifacts."""

    def __init__(
        self,
        metadata_repo: MetadataRepository,
        group_assigner: GroupAssigner,
        trainer: Trainer,
    ) -> None:
        self._metadata_repo = metadata_repo
        self._group_assigner = group_assigner
        self._trainer = trainer

    def run(self, cfg: ExperimentConfig, base_dir: Path) -> Dict[str, float]:
        out_dir = base_dir / cfg.tag()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save config so each run is reproducible
        with open(out_dir / "config.json", "w") as f:
            json.dump(
                {
                    "data": asdict(cfg.data),
                    "model": asdict(cfg.model),
                    "cv": asdict(cfg.cv),
                },
                f,
                indent=2,
            )

        df, md_raw, labels, feature_cols = self._metadata_repo.load()
        md_dim = len(feature_cols)

        # Prepare groups only if needed
        #groups: Optional[np.ndarray] = None
        #if cfg.cv.method == "gkf":
        #    if cfg.cv.n_clusters is None:
        #        raise ValueError("cv.n_clusters is required for GroupKFold.")
        #    if cfg.cv.n_clusters < cfg.cv.n_splits:
        #        # Skip logic: cannot split into more folds than groups
        #        raise ValueError(f"Invalid config: n_clusters={cfg.cv.n_clusters} < n_splits={cfg.cv.n_splits}")
        #    groups = self._group_assigner.get_groups(df, cfg.cv.n_clusters)

        groups: Optional[np.ndarray] = None
        if cfg.cv.method == "gkf":
            groups_path = DATA_DIR / "cache" / "groups_seqid.npy"
            if not groups_path.exists():
                raise FileNotFoundError(
                    f"MMseqs groups file not found: {groups_path}\n"
                    f"Run your mmseqs script first to create it."
                )

            groups = np.load(groups_path).astype(int)

            if len(groups) != len(df):
                raise ValueError(
                    f"groups length mismatch: len(groups)={len(groups)} vs len(df)={len(df)}"
                )

            n_groups = len(np.unique(groups))
            if n_groups < cfg.cv.n_splits:
                raise ValueError(
                    f"Not enough groups for GroupKFold: have {n_groups} groups but n_splits={cfg.cv.n_splits}"
                )

            print(f"[GKF] Using MMseqs groups from {groups_path} | n_groups={n_groups} | n_samples={len(groups)}")

        split_strategy = make_split_strategy(cfg.cv)

        # Store predictions for aggregated CM and summary
        all_y_true: List[int] = []
        all_y_pred: List[int] = []
        fold_metrics: List[FoldMetrics] = []

        graph_builder = GraphBuilder(EMB_PATH, ADJ_PATH, cfg.data.cut_off)

        for fold_id, tr_idx, te_idx in split_strategy.splits(md_raw, labels, groups=groups):
            # Split outer-train again into inner-train + inner-val
            inner_tr_idx, val_idx = self._make_inner_split(
                tr_idx=tr_idx,
                labels=labels,
                groups=groups,
                seed=cfg.cv.seed + fold_id,
                val_size=0.2,
            )

            # Fit scaler only on inner training data
            scaler = StandardScaler().fit(md_raw[inner_tr_idx])

            tr_graphs = graph_builder.build(df, md_raw, inner_tr_idx, scaler)
            val_graphs = graph_builder.build(df, md_raw, val_idx, scaler)
            te_graphs = graph_builder.build(df, md_raw, te_idx, scaler)

            # Train on inner-train, early-stop on inner-val, report on outer-test
            _, fold_row, _ = self._trainer.train_fold(
                md_dim=md_dim,
                model_cfg=cfg.model,
                train_graphs=tr_graphs,
                val_graphs=val_graphs,
                test_graphs=te_graphs,
                out_dir=out_dir,
                fold_id=fold_id,
            )


            # We rebuild y_true/y_pred from fold CM file? Better: compute here again quickly.
            # But the trainer already computed y_true/y_pred. To keep things simple, we compute aggregated CM from fold_row only?
            # We want real aggregated CM, so we do a lightweight eval here:
            y_true, y_pred = self._predict_labels(
                md_dim=md_dim,
                model_cfg=cfg.model,
                model_path=out_dir / f"model_fold{fold_id}.pt",
                graphs=te_graphs,
            )

            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

            fold_metrics.append(
                FoldMetrics(
                    fold=fold_id,
                    accuracy=float(fold_row["accuracy"]),
                    f1=float(fold_row["f1"]),
                    roc_auc=float(fold_row["roc_auc"]),
                    pr_auc=float(fold_row["pr_auc"]),
                )
            )

        # Aggregated CM
        self._plot_cm(all_y_true, all_y_pred, out_dir / "conf_matrix_all_folds.png", dpi=700)

        # Aggregated summary (use simple micro metrics across all predictions)
        summary = {
            "accuracy": float(accuracy_score(all_y_true, all_y_pred)),
            "precision": float(precision_score(all_y_true, all_y_pred, zero_division=0)),
            "recall": float(recall_score(all_y_true, all_y_pred, zero_division=0)),
            "f1": float(f1_score(all_y_true, all_y_pred, zero_division=0)),
            "n_samples": int(len(all_y_true)),
            "timestamp": time.time(),
        }

        # Add fold stats (mean/std) for quick comparison
        f1s = np.array([m.f1 for m in fold_metrics], dtype=float)
        accs = np.array([m.accuracy for m in fold_metrics], dtype=float)
        summary["folds"] = {
            "mean_f1": float(np.nanmean(f1s)),
            "std_f1": float(np.nanstd(f1s)),
            "mean_acc": float(np.nanmean(accs)),
            "std_acc": float(np.nanstd(accs)),
        }

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    @staticmethod
    def _plot_cm(y_true: List[int], y_pred: List[int], out_path: Path, dpi: int = 300) -> None:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["unstable", "stable"])
        disp.plot(cmap="Blues")
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi)
        plt.close()

    @staticmethod
    def _predict_labels(
        md_dim: int,
        model_cfg: ModelConfig,
        model_path: Path,
        graphs: List[Data],
    ) -> Tuple[List[int], List[int]]:
        """Loads saved weights and predicts labels for graphs."""
        loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
        model = GATI(md_dim=md_dim, cfg=model_cfg).to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        y_true: List[int] = []
        y_pred: List[int] = []
        with torch.no_grad():
            for b in loader:
                b = b.to(DEVICE)
                logits = model(b)
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                true = b.y.cpu().numpy().tolist()
                y_true.extend(true)
                y_pred.extend(pred)
        return y_true, y_pred

    @staticmethod
    def _make_inner_split(
        tr_idx: np.ndarray,
        labels: np.ndarray,
        groups: Optional[np.ndarray],
        seed: int,
        val_size: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the outer-train indices into:
        - inner_train
        - inner_val

        For grouped CV we also keep groups separated in the inner split.
        """
        if groups is not None:
            splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
            inner_tr_pos, inner_val_pos = next(
                splitter.split(tr_idx, labels[tr_idx], groups=groups[tr_idx])
            )
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
            inner_tr_pos, inner_val_pos = next(
                splitter.split(tr_idx, labels[tr_idx])
            )

        inner_tr_idx = tr_idx[inner_tr_pos]
        inner_val_idx = tr_idx[inner_val_pos]
        return inner_tr_idx, inner_val_idx

# =============================================================================
# Selection utilities (Single Responsibility: choose best run)
# =============================================================================

def find_best_run(base_dir: Path, metric: str = "folds.mean_f1") -> Optional[Path]:
    """
    Scan all summary.json files and pick the best run by a metric path.

    metric path examples:
      - "f1" (micro f1 across all folds)
      - "folds.mean_f1"
      - "folds.mean_acc"
    """
    summaries = list(base_dir.glob("**/summary.json"))
    if not summaries:
        return None

    def get_metric(summary: dict) -> float:
        cur = summary
        for part in metric.split("."):
            cur = cur.get(part, None) if isinstance(cur, dict) else None
        return float(cur) if cur is not None else float("-inf")

    best_path = None
    best_val = float("-inf")

    for s in summaries:
        try:
            with open(s, "r") as f:
                summary = json.load(f)
            val = get_metric(summary)
            if np.isnan(val):
                continue
            if val > best_val:
                best_val = val
                best_path = s.parent
        except Exception:
            continue

    return best_path


# =============================================================================
# Main: define grids and run
# =============================================================================

def run_all_experiments() -> None:
    """
    Main entry point.

    You can keep the grid small at first to test the framework.
    Then enlarge it when everything works.
    """
    metadata_repo = MetadataRepository(CSV_PATH)
    group_assigner = GroupAssigner(EMB_PATH, CACHE_DIR)
    trainer = Trainer(lr=LEARNING_RATE, epochs=EPOCHS, patience=PATIENCE)
    runner = ExperimentRunner(metadata_repo, group_assigner, trainer)

    # Small example grid (expand as you want)
    cut_offs = [0.8, 0.5, 0.2]
    layers_options = [2, 3]
    hidden_dims = [126, 256]
    dropout_rates = [0.3]

    # CV methods to compare
    cv_methods = [
        CVConfig(method="sss", n_splits=10, test_size=0.2, seed=42),
        CVConfig(method="skf", n_splits=10, seed=42),
        CVConfig(method="gkf", n_splits=10, seed=42),
    ]

    # If you want to compare more group settings, add more CVConfig lines:
    # CVConfig(method="gkf", n_splits=7, seed=42, n_clusters=10),

    for (cut_off, num_layers, hidden_dim, dropout_rate, cv_cfg) in product(
        cut_offs, layers_options, hidden_dims, dropout_rates, cv_methods
    ):
        cfg = ExperimentConfig(
            data=DataConfig(cut_off=cut_off),
            model=ModelConfig(hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate),
            cv=cv_cfg,
        )

        out_dir = RUNS_DIR / cv_cfg.method
        out_dir.mkdir(exist_ok=True)

        run_folder = out_dir / cfg.tag()
        if run_folder.exists() and (run_folder / "summary.json").exists():
            print(f"Skip (already done): {run_folder}")
            continue

        print(f"\n=== RUN: {cv_cfg.method} | {cfg.tag()} ===")

        try:
            summary = runner.run(cfg, out_dir)
            print(f"Done. mean_f1={summary['folds']['mean_f1']:.4f}  micro_f1={summary['f1']:.4f}")
        except Exception as e:
            # Do not crash the full grid; log and continue.
            print(f"FAILED: {cfg.tag()}\n  Reason: {e}")

    # Pick best run per CV method
    for method in ["sss", "skf", "gkf"]:
        method_dir = RUNS_DIR / method
        best = find_best_run(method_dir, metric="folds.mean_f1")
        if best is None:
            print(f"\nNo runs found for {method}.")
        else:
            with open(best / "summary.json", "r") as f:
                s = json.load(f)
            print(
                f"\nBest for {method}: {best.name}\n"
                f"  mean_f1={s['folds']['mean_f1']:.4f}  std_f1={s['folds']['std_f1']:.4f}\n"
                f"  micro_f1={s['f1']:.4f}  acc={s['accuracy']:.4f}"
            )

    # Optional: pick overall best among all methods
    best_overall = find_best_run(RUNS_DIR, metric="folds.mean_f1")
    if best_overall:
        with open(best_overall / "summary.json", "r") as f:
            s = json.load(f)
        print(
            f"\nBEST OVERALL: {best_overall}\n"
            f"  mean_f1={s['folds']['mean_f1']:.4f}  std_f1={s['folds']['std_f1']:.4f}\n"
            f"  micro_f1={s['f1']:.4f}  acc={s['accuracy']:.4f}"
        )


def main() -> None:
    run_all_experiments()


if __name__ == "__main__":
    main()


