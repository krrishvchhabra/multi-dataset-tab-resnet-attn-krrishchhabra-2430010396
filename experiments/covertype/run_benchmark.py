import os
import sys
import time
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow imports from src/
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")
    )
)

from src.models.mlp import MLP
from src.models.tab_resnet_attn import TabResNetAttn
from src.utils import set_seed


DATA_PATH = "data/covtype.data"
OUTPUT_DIR = "results/covertype"

SEEDS = [42, 52, 62]
TRAIN_ROWS = 20_000
TEST_ROWS = 30_000
EPOCHS = 5
BATCH_SIZE = 512


def load_data():
    df = pd.read_csv(DATA_PATH, header=None)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    # Covertype classes are 1..7.
    y = y - 1

    return X, y


def train_torch_model(
    model,
    X_train,
    y_train,
    epochs,
    device,
):
    model.to(device)

    X_tensor = torch.tensor(
        X_train, dtype=torch.float32
    )
    y_tensor = torch.tensor(
        y_train, dtype=torch.long
    )

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = loss_fn(logits, yb)

            loss.backward()
            optimizer.step()

    return model


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(
            y_test, pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_test, pred,
            average="macro",
            zero_division=0,
        ),
    }

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_test)

        try:
            results["macro_ovr_auc"] = roc_auc_score(
                y_test,
                prob,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            results["macro_ovr_auc"] = np.nan

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X, y = load_data()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    all_rows = []

    for seed in SEEDS:
        set_seed(seed)

        train_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )

        # Compute-conscious subset of the training pool.
        train_idx = train_test_split(
            train_idx,
            train_size=TRAIN_ROWS,
            random_state=seed,
            stratify=y[train_idx],
        )[0]

        # Compute-conscious subset of the held-out pool.
        test_idx = train_test_split(
            test_idx,
            train_size=TEST_ROWS,
            random_state=seed,
            stratify=y[test_idx],
        )[0]

        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(
            X_train
        ).astype(np.float32)

        X_test_scaled = scaler.transform(
            X_test
        ).astype(np.float32)

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                random_state=seed,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=seed,
                n_jobs=4,
            ),
        }

        for name, model in models.items():
            start = time.time()

            model.fit(X_train, y_train)

            elapsed = time.time() - start

            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)

            row = {
                "dataset": "Covertype",
                "model": name,
                "seed": seed,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "accuracy": accuracy_score(
                    y_test, pred
                ),
                "macro_f1": f1_score(
                    y_test, pred,
                    average="macro",
                    zero_division=0,
                ),
                "macro_recall": recall_score(
                    y_test, pred,
                    average="macro",
                    zero_division=0,
                ),
                "macro_ovr_auc": roc_auc_score(
                    y_test,
                    prob,
                    multi_class="ovr",
                    average="macro",
                ),
                "training_seconds": elapsed,
            }

            all_rows.append(row)

        torch_models = {
            "MLP": MLP(
                input_dim=X_train.shape[1],
                num_classes=7,
                hidden_dims=(128, 64),
            ),
            "Tab-ResNet-Attn": TabResNetAttn(
                num_features=X_train.shape[1],
                num_classes=7,
                token_dim=16,
                num_heads=4,
                hidden_dim=64,
                num_residual_blocks=2,
            ),
        }

        for name, model in torch_models.items():
            start = time.time()

            model = train_torch_model(
                model,
                X_train_scaled,
                y_train,
                EPOCHS,
                device,
            )

            elapsed = time.time() - start

            model.eval()

            with torch.no_grad():
                logits = model(
                    torch.tensor(
                        X_test_scaled,
                        dtype=torch.float32,
                        device=device,
                    )
                )

                prob = torch.softmax(
                    logits, dim=1
                ).cpu().numpy()

                pred = prob.argmax(axis=1)

            row = {
                "dataset": "Covertype",
                "model": name,
                "seed": seed,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "accuracy": accuracy_score(
                    y_test, pred
                ),
                "macro_f1": f1_score(
                    y_test, pred,
                    average="macro",
                    zero_division=0,
                ),
                "macro_recall": recall_score(
                    y_test, pred,
                    average="macro",
                    zero_division=0,
                ),
                "macro_ovr_auc": roc_auc_score(
                    y_test,
                    prob,
                    multi_class="ovr",
                ),
                "training_seconds": elapsed,
            }

            all_rows.append(row)

    runs = pd.DataFrame(all_rows)

    summary = (
        runs.groupby("model")
        .agg({
            "accuracy": ["mean", "std"],
            "macro_f1": ["mean", "std"],
            "macro_recall": ["mean", "std"],
            "macro_ovr_auc": ["mean", "std"],
            "training_seconds": ["mean", "std"],
        })
    )

    runs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "benchmark_runs.csv",
        ),
        index=False,
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "benchmark_summary.csv",
        ),
    )

    with open(
        os.path.join(OUTPUT_DIR, "metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump({
            "dataset": "Covertype",
            "seeds": SEEDS,
            "train_rows": TRAIN_ROWS,
            "test_rows": TEST_ROWS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        }, f, indent=2)

    print(summary)


if __name__ == "__main__":
    main()
